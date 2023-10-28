#!/usr/bin/env python
# coding: utf-8

import os
import os.path as osp
import datetime
import time
import pdb
import tkinter

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
            return device
    else:
        raise Exception(f'No device found! Please connect a device and run '
                        f'the example again.')

''' カメラの撮影プログラム '''
# capture a single image.
def capture(device):
    # print(device.nodemap['AcquisitionMode'])
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
    return rgb_arr

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

# GUI
class Application(tkinter.Frame):
    def __init__(self, root, polar_cam_dev, polar_nodemap, save_root='data'):
        super().__init__(root, width=380, height=280, borderwidth=4, relief='groove')
        # camera object
        self.device_polar = polar_cam_dev
        self.nodemap_polar = polar_nodemap
        self.save_root = save_root
        # GUI object
        self.pack_propagate(0)
        self.root = root
        self.create_widgets()
        # count object
        # self.cap_count = 0
        
    def create_widgets(self):
        # define widgets
        cap_btn = tkinter.Button(self.root, text='撮影', command=self.capture)
        label = tkinter.Label(self.root, text='Mold type:')
        self.text_box = tkinter.Entry(self.root, width=30)
        # asign widgets
        label.grid(row=0, column=0, padx=(10, 5), pady=(10, 10))
        self.text_box.grid(row=0, column=1)
        cap_btn.grid(row=1, column=1)
        
    def capture(self):
        self.save_dir = self.text_box.get()
        filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".jpg"
        # Capture images
        start_time = time.time()
        self.nodemap_polar['PixelFormat'].value = "BayerRG8"
        bayerRG8_raw = capture(self.device_polar)
        self.nodemap_polar['PixelFormat'].value = "PolarizedAolp_BayerRG8"
        aolp_raw = capture(self.device_polar)
        self.nodemap_polar['PixelFormat'].value = "PolarizedDolp_BayerRG8"
        dolp_raw = capture(self.device_polar)
        end_time = time.time()
        # Post Processing
        rgb = post_process_RGB(bayerRG8_raw)
        aolp = post_process_AoLP(aolp_raw)
        dolp = post_process_DoLP(dolp_raw)
        # Setting savepath
        save_dir = os.path.join(self.save_root, self.save_dir)
        os.makedirs(os.path.join(save_dir, "image"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "aolp"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "dolp"), exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, "image", filename), rgb)
        cv2.imwrite(os.path.join(save_dir, "aolp", filename), aolp)
        cv2.imwrite(os.path.join(save_dir, "dolp", filename), dolp)
        print(f"Captured RGB, AoLP, and DoLP. Shooting Time: {end_time - start_time}[s]")

def main():
    device = device_set()
    ''' 撮影に関する設定 '''
    nodemap = device.nodemap
    nodemap['AcquisitionMode'].value = 'SingleFrame'
    ''' ストリーミングに関する設定 '''
    node_stream = device.tl_stream_nodemap
    node_stream['StreamAutoNegotiatePacketSize'].value = True
    node_stream['StreamPacketResendEnable'].value = True
    print("RGB Polarize")
    print("FrameRate:", nodemap['AcquisitionFrameRate'].value)
    print("ExposureTime:", nodemap['ExposureTime'].value)
    print("Ready to go!")
    root = tkinter.Tk()
    root.title('同時撮影')
    root.geometry('300x100')
    app = Application(root=root, polar_cam_dev=device, polar_nodemap=nodemap, save_root='data')
    app.mainloop()

    # device reset
    system.destroy_device()
    print('Destroyed all created devices')

if __name__ == "__main__":
    main()