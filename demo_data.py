#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:33:45 2019

@author: fengy
"""

import sys
import os

import numpy as np

from PIL import Image

import xml.etree.ElementTree as ET
import glob




def write_to_file(input_path,output_path,vid_list):
    fname=os.path.join(output_path,"wider_face_demo_filelist.txt")
    print("Writing to {}".format(fname))
    for video in vid_list:

        path1=video.split('/')
        with open(fname,"a+") as f:
            f.write(os.path.join(path1[-2],path1[-1] + " \n"))
            print(os.path.join(path1[-2],path1[-1] + " \n"))






if __name__ == "__main__":

    output_path='/home/fengy/Documents/tiny-faces-pytorch/data/WIDER/wider_face_split'
    #input_path='/home/fengy/Downloads/ILSVRC2015_VID/ILSVRC2015/Data/DET/train/ILSVRC2013_train/n00007846'
    input_path='/home/fengy/Documents/tiny-faces-pytorch/data/WIDER/WIDER_demo/images/AVA_train_0'

    
    videos = sorted(glob.glob(os.path.join(input_path, "*.png")))

   

    write_to_file(input_path,output_path,videos[21:50])
