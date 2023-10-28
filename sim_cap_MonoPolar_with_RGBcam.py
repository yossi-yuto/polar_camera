#!/usr/bin/env python
# coding: utf-8

import os
import os.path as osp
import datetime
import time
import pdb
import threading
import tkinter

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from arena_api import enums
from arena_api.system import system
from arena_api.buffer import BufferFactory
from arena_api.__future__.save import Writer
from multiprocessing import Value


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


def capture(device):
    # print(device.nodemap['AcquisitionMode'])
    with device.start_stream():
        buffer = device.get_buffer()
        np_array = np.asarray(buffer.data, dtype=np.uint8)
        buffer_bytes_per_pixel = int(len(buffer.data)/(buffer.width * buffer.height))
        np_array_reshaped = np_array.reshape(buffer.height, buffer.width, buffer_bytes_per_pixel)
        device.requeue_buffer(buffer)
    return np_array_reshaped

# scale adjustment
def scale_linear_bycolumn(rawpoints, high=255.0, low=0.0):
	mins = np.min(rawpoints, axis=0)
	maxs = np.max(rawpoints, axis=0)
	rng = maxs - mins
	return high - (((high - low) * (maxs - rawpoints)) / rng)

def post_process_AoLP(AoLP_raw) -> np.ndarray:
    _ = scale_linear_bycolumn(AoLP_raw).astype(np.uint8)
    nparray_reshaped = cv2.applyColorMap(_, cv2.COLORMAP_HSV)
    return nparray_reshaped

def post_process_DoLP(DoLP_raw) -> np.ndarray:
    nparray_reshaped = cv2.applyColorMap(DoLP_raw, cv2.COLORMAP_JET)
    return nparray_reshaped

# GUI
class Application(tkinter.Frame):
    def __init__(self, root, polar_cam_dev, color_cam_dev, polar_nodemap, color_nodemap, save_root='data'):
        super().__init__(root, width=380, height=280, borderwidth=4, relief='groove')
        
        # camera object
        self.device_polar = polar_cam_dev
        self.device_color = color_cam_dev
        self.nodemap_polar = polar_nodemap
        self.nodemap_color = color_nodemap
        self.save_root = save_root

        self.pack_propagate(0)
        self.root = root
        self.create_widgets()
        
        
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

        start_time = time.time()
        self.nodemap_polar['PixelFormat'].value = "Mono8"
        mono_raw = capture(self.device_polar)
        self.nodemap_polar['PixelFormat'].value = "PolarizedAolp_Mono8"
        AoLP_raw = capture(self.device_polar)
        self.nodemap_polar['PixelFormat'].value = "PolarizedDolp_Mono8"
        DoLP_raw = capture(self.device_polar)
        self.nodemap_color['PixelFormat'].value = "RGB8"
        rgb_raw = capture(self.device_color)
        end_time = time.time()
        
        # Post-Processing
        aolp = post_process_AoLP(AoLP_raw)
        dolp = post_process_DoLP(DoLP_raw)

        # Setting savepath
        save_dir = os.path.join(self.save_root, self.save_dir)
        os.makedirs(os.path.join(save_dir, "image"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "aolp"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "dolp"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "mono"), exist_ok=True)
        
        cv2.imwrite(os.path.join(save_dir, "image", filename), rgb_raw)
        cv2.imwrite(os.path.join(save_dir, "aolp", filename), aolp)
        cv2.imwrite(os.path.join(save_dir, "dolp", filename), dolp)
        cv2.imwrite(os.path.join(save_dir, "mono", filename), mono_raw)
        print(f"Captured image, aolp, dolp, and mono, shooting time: {end_time - start_time}[s]")


def main():
    # Set cameras (polarize and web)
    print("Recognize devices.")
    devices = device_set()
    
    # polarize device configuration
    nodemap_1 = devices[0].nodemap
    nodemap_2 = devices[1].nodemap
    if (nodemap_1['DeviceModelName'].value == "TRI050S-C") and (nodemap_2['DeviceModelName'].value == "TRI050S1-P"):
        device_color = devices[0]
        device_polar = devices[1] 
    else:
        device_color = devices[1]
        device_polar = devices[0] 
        
    # Device configuration
    
    # color camera
    color_node = device_color.nodemap
    color_node['AcquisitionMode'].value = 'SingleFrame'
    if color_node['PixelFormat'].value != 'BGR8':
        color_node['PixelFormat'].value = 'BGR8'

    color_node_stream = device_color.tl_stream_nodemap
    color_node_stream['StreamAutoNegotiatePacketSize'].value = True
    color_node_stream['StreamPacketResendEnable'].value = True

    # polar camera
    polar_node = device_polar.nodemap
    polar_node['AcquisitionMode'].value = 'SingleFrame'
    
    polar_node_stream = device_polar.tl_stream_nodemap
    polar_node_stream['StreamAutoNegotiatePacketSize'].value = True
    polar_node_stream['StreamPacketResendEnable'].value = True
    
    print("RGB Polarize")
    print("FrameRate:", color_node['AcquisitionFrameRate'].value, polar_node['AcquisitionFrameRate'].value)
    print("ExposureTime:", color_node['ExposureTime'].value, polar_node['ExposureTime'].value)
    
    #GUI setting
    print("Ready to go!")
    root = tkinter.Tk()
    root.title('同時撮影')
    root.geometry('300x100')
    app = Application(root=root, polar_cam_dev=device_polar, color_cam_dev=device_color, polar_nodemap=polar_node, color_nodemap=color_node, save_root='data')
    app.mainloop()

    # device reset
    system.destroy_device()
    print('Destroyed all created devices')

if __name__ == "__main__":
    main()