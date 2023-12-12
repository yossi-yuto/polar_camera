#!/usr/bin/env python
# coding: utf-8

import os
import os.path as osp
import datetime
import time
import pdb
import threading
import tkinter
from tkinter import ttk
from tkinter import font
import time

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import skimage.util

from arena_api import enums
from arena_api.system import system
from arena_api.buffer import BufferFactory
from arena_api.__future__.save import Writer
from multiprocessing import Value

from camera_config import *

# device recognize
"""
This function waits for the user to connect a device before raising
an exception
"""
def device_set():
    tries = 0
    tries_max = 6
    sleep_time_secs = 10
    while tries < tries_max:  # Wait for device for 60 seconds
        devices = system.create_device()
        if not devices:
            print(
                f'Try {tries+1} of {tries_max}: waiting for {sleep_time_secs} '
                f'secs for a device to be connected!')
            for sec_count in range(sleep_time_secs):
                time.sleep(1)
                print(f'{sec_count + 1 } seconds passed ',
                    '.' * sec_count, end='\r')
            tries += 1
        else:
            print(f'Created {len(devices)} device(s)\n')
            device = devices[0]
            return devices
    else:
        raise Exception(f'No device found! Please connect a device and run '
                        f'the example again.')


def capture_RGB(device):
    with device.start_stream():
        buffer = device.get_buffer()
        np_array = np.asarray(buffer.data, dtype=np.uint8)
        buffer_bytes_per_pixel = int(len(buffer.data)/(buffer.width * buffer.height))
        np_array_reshaped = np_array.reshape(buffer.height, buffer.width, buffer_bytes_per_pixel)
        device.requeue_buffer(buffer)
    return np_array_reshaped, device.nodemap['ExposureTime'].value

def capture(device):
    # print(device.nodemap['ExposureTime'].value)
    with device.start_stream():
        buffer = device.get_buffer()
        np_array = np.asarray(buffer.data, dtype=np.uint8)
        buffer_bytes_per_pixel = int(len(buffer.data)/(buffer.width * buffer.height))
        np_array_reshaped = np_array.reshape(buffer.height, buffer.width, buffer_bytes_per_pixel)
        device.requeue_buffer(buffer)
    return np_array_reshaped

''' 撮影画像の後処理に関するプログラム '''
# Return 2x2 grid image. (Red, Green1, Green2, Blue)
def convert_polarize_2x2_grid(polarize_raw_image):
    h, w = polarize_raw_image.shape
    original_shape = (w, h)
    red = cv2.resize(polarize_raw_image[::2, ::2] , original_shape)
    green1 = cv2.resize(polarize_raw_image[::2, 1::2] , original_shape)
    green2 = cv2.resize(polarize_raw_image[1::2, ::2] , original_shape)
    blue = cv2.resize(polarize_raw_image[1::2, 1::2] , original_shape)
    grid2x2 = cv2.vconcat([cv2.hconcat([red, green1]), cv2.hconcat([green2, blue])])
    return grid2x2

# scale adjustment
def scale_linear_bycolumn(rawpoints, high=255.0, low=0.0):
	mins = np.min(rawpoints, axis=0)
	maxs = np.max(rawpoints, axis=0)
	rng = maxs - mins
	return high - (((high - low) * (maxs - rawpoints)) / rng)

# Return RGB image(Red, Green1, Blue) from bayerRG8raw.
def bayerArr_to_RGBArr(bayerRG_arr):
    h, w = bayerRG_arr.shape[0] // 2, bayerRG_arr.shape[1] // 2
    rgb_arr = np.zeros((h, w, 3))
    blocks = skimage.util.view_as_blocks(bayerRG_arr, (2, 2))
    rgb_arr[:,:,0] = np.reshape(blocks[::2, ::2].transpose(0,2,1,3), (h, w))
    rgb_arr[:,:,1] = np.reshape(blocks[::2, 1::2].transpose(0,2,1,3), (h, w))
    rgb_arr[:,:,2] = np.reshape(blocks[1::2, 1::2].transpose(0,2,1,3), (h, w))
    return np.uint8(rgb_arr)

# Return 2x2 AoLP images
def post_process_AoLP(AoLP_raw) -> np.ndarray:
    _ = scale_linear_bycolumn(AoLP_raw).astype(np.uint8)
    _ = convert_polarize_2x2_grid(_)
    AoLP_grid2x2 = cv2.applyColorMap(_, cv2.COLORMAP_HSV)
    return AoLP_grid2x2
# Return 2x2 DoLP images
def post_process_DoLP(DoLP_raw) -> np.ndarray:
    _ = convert_polarize_2x2_grid(DoLP_raw)
    DoLP_grid2x2 = cv2.applyColorMap(_, cv2.COLORMAP_JET)
    return DoLP_grid2x2
# Return RGB image. (pixel shape is half)
def post_process_RGB(bayerRG_raw) -> np.ndarray:
    rgb_img = bayerArr_to_RGBArr(bayerRG_raw)
    return rgb_img

def buffer_to_array(buffer) -> np.ndarray:
    np_array = np.asarray(buffer.data, dtype=np.uint8)
    buffer_bytes_per_pixel = int(len(buffer.data)/(buffer.width * buffer.height))
    np_array_reshaped = np_array.reshape(buffer.height, buffer.width, buffer_bytes_per_pixel)
    return np_array_reshaped

def calc_DoLP(S_0: np.ndarray, S1: np.ndarray, S2: np.ndarray) -> np.ndarray:
    top = np.sqrt((S1**2 + S2**2))
    under = S_0
    under[under == 0] = np.nan
    dolp = (top / under) * 255.
    dolp = np.nan_to_num(dolp).astype(np.uint8)
    return dolp

def calc_AoLP(S1: np.ndarray, S2: np.ndarray) -> np.ndarray:
    aolp = 0.5 * np.arctan2(S2, S1)
    aolp_agree =  np.degrees(aolp)
    aolp_agree[aolp_agree < 0] += 180
    n_aolp = ((aolp_agree / 180.) * 255.).astype(np.uint8)
    return n_aolp

def decompose(angle_img) -> tuple:
    red = angle_img[::2, ::2].astype(np.float32)
    green1 = angle_img[::2, 1::2].astype(np.float32)
    green2 = angle_img[1::2, ::2].astype(np.float32)
    blue = angle_img[1::2, 1::2].astype(np.float32)
    return red, green1, green2, blue


def concat2x2_image(img1: np.ndarray, img2: np.ndarray, img3: np.ndarray, img4: np.ndarray) -> np.ndarray:
    top_row = np.hstack((img1, img2))
    bottom_row = np.hstack((img3, img4))
    return np.vstack((top_row, bottom_row))


def calc_aolp_dolp(p0, p45, p90, p135) -> tuple:
    S0 = p0 + p90
    S1 = p0 - p90
    S2 = p45 - p135
    aolp = calc_AoLP(S1, S2)
    dolp = calc_DoLP(S0, S1, S2)
    return aolp, dolp

# GUI
class Application(tkinter.Frame):
    def __init__(self, root, polar_cam_dev, exposure_auto=True, gain_auto=True, save_root='data'):
        super().__init__(root, width=400, height=500, borderwidth=4, relief='groove')
        
        # camera object
        self.device_polar = polar_cam_dev
        self.nodemap_polar = self.device_polar.nodemap
        self.save_root = save_root
        self.ET_auto = exposure_auto

        self.pack_propagate(0)
        self.root = root
        self.create_widgets()
        self.update_var()
        
    def create_widgets(self):

        self.text_var_ET = tkinter.StringVar()
        # define widgets
        font_btn = font.Font(family="Helvatica", size=14, weight="bold")
        cap_btn = tkinter.Button(self.root, text='撮影', command=self.capture, width=20, height=5, bg="black", fg="white", font=font_btn)
        label_F = tkinter.Label(self.root, text='F値: ')
        self.text_box_F = tkinter.Entry(self.root, width=10)
        self.text_box_F.insert(0, "6.0")
        label_ET = tkinter.Label(self.root, text='Exposure Time [microsec]: ')
        self.text_box_ET = tkinter.Label(self.root, textvariable=self.text_var_ET)
        # self.text_box_ET.insert(0, )
        label_type = tkinter.Label(self.root, text='Mold type:')
        self.text_box_type = tkinter.Entry(self.root, width=30)
        self.text_box_type.insert(0, "tmp")
        separator = ttk.Separator(self.root, orient='horizontal')
        label_FR = tkinter.Label(self.root, text="FrameRate [Hz]: ")
        self.text_box_FR = tkinter.Entry(self.root, width=30)
        self.text_box_FR.insert(0, self.nodemap_polar['AcquisitionFrameRate'].value)
        self.label_ETauto = tkinter.Label(self.root, text=f"ExposureTimeAuto: ")
        self.label_ETauto_state = tkinter.Label(self.root, text=self.nodemap_polar['ExposureAuto'].value)
        
        # asign widgets
        label_type.grid(row=0, column=0, padx=(10,5), pady=(10,10))
        self.text_box_type.grid(row=0, column=1, pady=(10,10))
        
        cap_btn.grid(row=1, column=1, pady=(10,10))
        
        separator.grid(row=2, columnspan=3, sticky='ew', pady=(10,5))
        
        label_F.grid(row=3, column=0, padx=(10,5), pady=(10,8))
        self.text_box_F.grid(row=3, column=1, pady=(10,8))
        
        label_ET.grid(row=4, column=0, padx=(10,5), pady=(8,8))
        self.text_box_ET.grid(row=4, column=1, pady=(8,8))
        
        label_FR.grid(row=5, column=0, padx=(10,5), pady=(8,8))
        self.text_box_FR.grid(row=5, column=1, pady=(8,8))
        
        self.label_ETauto.grid(row=6, column=0, padx=(10,5), pady=(8,8))
        self.label_ETauto_state.grid(row=6, column=1, padx=(10,5), pady=(8,8))

    def capture(self):
        aperture = self.text_box_F.get()
        self.save_dir = self.text_box_type.get()
        # exposure_time = float(self.text_box_ET.get())
        frameRate = float(self.text_box_FR.get())
        # filename and framerate setting
        self.nodemap_polar['AcquisitionFrameRate'].value = frameRate
        
        capture_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        framerate = self.nodemap_polar['AcquisitionFrameRate'].value 
        
        # capture
        print(self.nodemap_polar['ExposureTime'].value)
        
        start_time = time.time()
        self.nodemap_polar['PixelFormat'].value = "PolarizedAngles_0d_45d_90d_135d_BayerRG8"
        with self.device_polar.start_stream(1):
            image = self.device_polar.get_buffer()
            buffer_rgb = BufferFactory.copy(image) 
            self.device_polar.requeue_buffer(image)
        print("Finish capture.")

        end_time = time.time()
        Stokes_array = buffer_to_array(buffer_rgb)
        # bayerArr_to_RGBArr(Stokes_array)
        BufferFactory.destroy(buffer_rgb)
        # Post-Processing
        p_0= Stokes_array[:,:,0]
        p_45 = Stokes_array[:,:,1]
        p_90 = Stokes_array[:,:,2]
        p_135 = Stokes_array[:,:,3]
        
        tmp = np.zeros((p_0.shape[0] * 2, p_0.shape[1] * 2), dtype=np.uint8)
        tmp[1::2, 1::2] = p_0
        tmp[::2, 1::2] = p_45
        tmp[::2, ::2] = p_90
        tmp[1::2, ::2] = p_135

        # split angles
        p0_red, p0_g1, p0_g2, p0_b = decompose(p_0)
        p45_red, p45_g1, p45_g2, p45_b = decompose(p_45)
        p90_red, p90_g1, p90_g2, p90_b = decompose(p_90)
        p135_red, p135_g1, p135_g2, p135_b = decompose(p_135)
        
        aolp_R, dolp_R = calc_aolp_dolp(p0_red, p45_red, p90_red, p135_red)
        aolp_G1, dolp_G1 = calc_aolp_dolp(p0_g1, p45_g1, p90_g1, p135_g1)
        aolp_B, dolp_B = calc_aolp_dolp(p0_b, p45_b, p90_b, p135_b)
        aolp_G2, dolp_G2 = calc_aolp_dolp(p0_g2, p45_g2, p90_g2, p135_g2)
        
        rgb = bayerArr_to_RGBArr(tmp)
        dolp = concat2x2_image(dolp_R, dolp_G1, dolp_G2, dolp_B)
        aolp = concat2x2_image(aolp_R, aolp_G1, aolp_G2, aolp_B)
        # Set exposuretime automation.
        print(f"Captured image, aolp, dolp, and mono, shooting time: {end_time - start_time}[s]")

        # Setting savepath
        save_dir = os.path.join(self.save_root, self.save_dir)
        os.makedirs(os.path.join(save_dir, "RGB"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "aolp"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "dolp"), exist_ok=True)
        
        filename = "F-"+ str(aperture)+ "_FR-" + f"{framerate:.3f}" + f"_ET-{self.nodemap_polar['ExposureTime'].value:.0f}" + f"_G-{self.nodemap_polar['Gain'].value}"+ "__" + capture_time + ".jpg"
        cv2.imwrite(os.path.join(save_dir, "RGB", filename), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, "aolp", filename), cv2.applyColorMap(aolp, cv2.COLORMAP_HSV))
        cv2.imwrite(os.path.join(save_dir, "dolp", filename), cv2.applyColorMap(dolp, cv2.COLORMAP_JET))
        print("Save images. RGB, AoLP, DoLP")
        
    def update_var(self):
        current_exposure_time = self.nodemap_polar['ExposureTime'].value
        self.text_var_ET.set(current_exposure_time)
        self.root.after(500, self.update_var)


def main():
    # Set cameras (polarize and web)
    devices = device_set()
    exposure_auto = False
    gain_auto = False
    # devices = [setup_CameraDevice(device, exposure_auto, gain_auto, target_brightness=90) for device in devices]
    devices = [setup2_CameraDevice(device) for device in devices]
    for device in devices:
        if device.nodemap['DeviceModelName'].value == "TRI050S1-Q":
            device_polar = device
            break
        else:
            pass
    
    #GUI setting
    print("Ready to go!")
    root = tkinter.Tk()
    root.title('偏光カメラ撮影用UI')
    root.geometry('450x400')
    app = Application(root=root, polar_cam_dev=device_polar, exposure_auto=exposure_auto, save_root='data')
    app.mainloop()

    # device reset
    system.destroy_device()
    print('Destroyed all created devices')

if __name__ == "__main__":
    main()