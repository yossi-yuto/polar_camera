#!/usr/bin/env python
# coding: utf-8

import os
import os.path as osp
from pathlib import Path
import datetime
import time
import pdb
import tkinter
from tkinter import ttk
from tkinter import font
import tifffile as tiff

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import skimage.util

from arena_api import enums
from arena_api.system import system
from arena_api.buffer import BufferFactory
from arena_api.__future__.save import Writer

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
    print("buffer_bytes_per_pixel:", buffer_bytes_per_pixel)
    np_array_reshaped = np_array.reshape(buffer.height, buffer.width, buffer_bytes_per_pixel)
    print("np_array_reshaped shape:", np_array_reshaped.shape)
    return np_array_reshaped


''' AoLPの計算 '''
def calc_AoLP(S1: np.ndarray, S2: np.ndarray, DoLP: np.ndarray, noise_level=10, zeros_replace_color=120) -> np.ndarray:
    # 偏光度の計算（radian）
    aolp = 0.5 * np.arctan2(S2, S1)
    # 単位変換（radian -> degree）
    aolp_angle = np.degrees(aolp) 
    # マイナス値を18マイナスして0-180度に変換
    aolp_angle[aolp_angle < 0] += 180
    return aolp_angle   # (0-180) degree


''' DoLPの計算 '''
def calc_DoLP(S_0: np.ndarray, S1: np.ndarray, S2: np.ndarray) -> np.ndarray:
    # 偏光度の計算
    dolp = np.sqrt(S1**2 + S2**2) / np.where(S_0 == 0, 1e-10, S_0)
    # noise 除去
    dolp = np.clip(dolp, 0, 1)
    # print(dolp.min(), dolp.max())
    # noise 確認
    # if dolp.max() > 1.01:
    #     print("DoLP over 1.0", np.max(dolp), "index :", np.unravel_index(np.argmax(dolp), dolp.shape))
    return dolp # (0-1) float



def calc_aolp_dolp(p0, p45, p90, p135) -> tuple:
    # S0 = p0 + p90
    # S1 = p0 - p90
    # S2 = p45 - p135
    p0_norm = p0 / 255.
    p45_norm = p45 / 255.
    p90_norm = p90 / 255.
    p135_norm = p135 / 255.
    S0_norm = (p0_norm + p90_norm + p45_norm + p135_norm) * 0.5
    S1_norm = p0_norm - p90_norm
    S2_norm = p45_norm - p135_norm
    dolp = calc_DoLP(S0_norm, S1_norm, S2_norm)
    aolp = calc_AoLP(S1_norm, S2_norm, dolp)
    return aolp, dolp


def vis_aolp_dolp(aolp: np.ndarray, dolp: np.ndarray) -> tuple:
    # visualize AoLP 
    hsv_image = np.zeros((aolp.shape[0], aolp.shape[1], 3), dtype=np.uint8)
    hsv_image[..., 0] = aolp.astype(np.uint8)
    hsv_image[..., 1] = 255
    hsv_image[..., 2] = 255
    hsv_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    # visualize DoLP
    heat_image = (dolp * 255).astype(np.uint8)
    heat_image = cv2.applyColorMap(heat_image, cv2.COLORMAP_JET)
    return hsv_image, heat_image


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

# GUI
class Application(tkinter.Frame):
    def __init__(self, root, polar_cam_dev, save_root='data'):
        super().__init__(root, width=400, height=500, borderwidth=4, relief='groove')
        
        # camera object
        self.device_polar = polar_cam_dev
        self.nodemap_polar = self.device_polar.nodemap
        self.save_root = Path(save_root)
        self.save_dir = None

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
        label_ET = tkinter.Label(self.root, text='Exposure Time [sec]: ')
        self.text_box_ET = tkinter.Label(self.root, textvariable=self.text_var_ET)
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
        # capture setting
        aperture = self.text_box_F.get()
        frameRate = float(self.text_box_FR.get())
        self.nodemap_polar['AcquisitionFrameRate'].value = frameRate
        capture_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        framerate = self.nodemap_polar['AcquisitionFrameRate'].value 
        # directory setting
        self.save_dir = self.save_root / self.text_box_type.get()
        (self.save_dir / "RGB").mkdir(parents=True, exist_ok=True)
        (self.save_dir / "aolp").mkdir(parents=True, exist_ok=True)
        (self.save_dir / "aolp_vis").mkdir(parents=True, exist_ok=True)
        (self.save_dir / "dolp").mkdir(parents=True, exist_ok=True)
        (self.save_dir / "dolp_vis").mkdir(parents=True, exist_ok=True)
        (self.save_dir / "aolp_crop").mkdir(parents=True, exist_ok=True)
        (self.save_dir / "dolp_crop").mkdir(parents=True, exist_ok=True)
        # capture処理
        for i, view_point in enumerate(['left', 'right']):
            print(self.nodemap_polar['ExposureTime'].value)
            with self.device_polar.start_stream():
                image = self.device_polar.get_buffer(timeout=1000)
                buffer_rgb = BufferFactory.copy(image) 
                self.device_polar.requeue_buffer(image)
                print("finish capture:", view_point)
            
                Stokes_array = buffer_to_array(buffer_rgb) #(H, W, 4) polarization 0, 45, 90, 135
                # BufferFactory.destroy(buffer_rgb)
                p_0= Stokes_array[:,:,0]
                p_45 = Stokes_array[:,:,1]
                p_90 = Stokes_array[:,:,2]
                p_135 = Stokes_array[:,:,3]
            
                bayerRG = np.zeros((p_0.shape[0] * 2, p_0.shape[1] * 2), dtype=np.uint8)
                bayerRG[1::2, 1::2] = p_0
                bayerRG[::2, 1::2] = p_45
                bayerRG[::2, ::2] = p_90
                bayerRG[1::2, ::2] = p_135

                p0_red, p0_g1, p0_g2, p0_b = decompose(p_0)
                p45_red, p45_g1, p45_g2, p45_b = decompose(p_45)
                p90_red, p90_g1, p90_g2, p90_b = decompose(p_90)
                p135_red, p135_g1, p135_g2, p135_b = decompose(p_135)
            
                aolp_R, dolp_R = calc_aolp_dolp(p0_red, p45_red, p90_red, p135_red)
                aolp_G1, dolp_G1 = calc_aolp_dolp(p0_g1, p45_g1, p90_g1, p135_g1)
                aolp_B, dolp_B = calc_aolp_dolp(p0_b, p45_b, p90_b, p135_b)
                aolp_G2, dolp_G2 = calc_aolp_dolp(p0_g2, p45_g2, p90_g2, p135_g2)
                
                rgb = bayerArr_to_RGBArr(bayerRG)
                dolp = concat2x2_image(dolp_R, dolp_G1, dolp_G2, dolp_B)
                aolp = concat2x2_image(aolp_R, aolp_G1, aolp_G2, aolp_B)
                
                vis_aolp, vis_dolp = vis_aolp_dolp(aolp, dolp)

                # color convert
                BGR_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                print("image size: ", BGR_img.shape)
                print("aolp size: ", aolp.shape)
                print("dolp size: ", dolp.shape)
                # ファイル名生成
                filename = f"F-{aperture}_FR-{framerate:.3f}_ET-{self.nodemap_polar['ExposureTime'].value:.0f}_G-{self.nodemap_polar['Gain'].value}_{capture_time}_{view_point}.jpg"
                # 画像保存処理
                cv2.imwrite(str(self.save_dir / "RGB" / filename), BGR_img)
                tiff.imwrite(str(self.save_dir / "aolp" / filename.replace(".jpg", ".tiff")), aolp)
                tiff.imwrite(str(self.save_dir / "dolp" / filename.replace(".jpg", ".tiff")), dolp)
                cv2.imwrite(str(self.save_dir / "aolp_vis" / filename), vis_aolp)
                cv2.imwrite(str(self.save_dir / "dolp_vis" / filename), vis_dolp)
                
                # 画像のクロップ保存
                w, h = vis_aolp.shape[1], vis_aolp.shape[0]
                cv2.imwrite(str(self.save_dir / "aolp_crop" / filename), vis_aolp[:h//2, :w//2])
                cv2.imwrite(str(self.save_dir / "dolp_crop" / filename), vis_dolp[:h//2, :w//2])

                # window plot
                vis_w, vis_h = w // 3, h // 3
                cv2.imshow("DoLP", cv2.resize(vis_dolp[:h//2, :w//2], (vis_w, vis_h)))
                cv2.imshow("AoLP", cv2.resize(vis_aolp[:h//2, :w//2], (vis_w, vis_h)))
                cv2.imshow("RGB", cv2.resize(bgr, (vis_w, vis_h)))
                cv2.moveWindow("RGB", 50, 0)
                cv2.moveWindow("AoLP", 60 + vis_w, 0)
                cv2.moveWindow("DoLP", 50 , 10+vis_h)
                cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def update_var(self):
        current_exposure_time = self.nodemap_polar['ExposureTime'].value * 1e-6
        self.text_var_ET.set(current_exposure_time)
        self.root.after(500, self.update_var)


def main():
    # Set cameras (polarize and web)
    devices = device_set()
    devices = [setup2_CameraDevice(device) for device in devices]
    device_polar = devices[0]
    # configure polarized camera
    device_polar.nodemap['PixelFormat'].value = 'PolarizedAngles_0d_45d_90d_135d_BayerRG8'
    device_polar.nodemap['Height'].value = device_polar.nodemap['Height'].max
    device_polar.nodemap['Width'].value = device_polar.nodemap['Width'].max
    
    #GUI setting
    print("Ready to go!")
    root = tkinter.Tk()
    root.title('偏光カメラ撮影用UI')
    root.geometry('450x400+600+100')
    app = Application(root=root, polar_cam_dev=device_polar, save_root='data')
    app.mainloop()

    # device reset
    system.destroy_device()
    print('Destroyed all created devices')

if __name__ == "__main__":
    main()