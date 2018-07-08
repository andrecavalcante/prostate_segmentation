# This script randomly separates MR exames into train, validation and test datasets. 
# For convenience, it converts the original .nrrd dataset files into .npy.
# And finally generates .cvs files for Pytorch Dataset and Dataloaders

from __future__ import print_function

import torch
import pickle
import numpy as np, gzip, glob, cv2
import scipy.misc
import random
import os

import SimpleITK as sitk

root_dir = ""      # project root directory
data_dir = ""      # PROMISE12 image directory
label_dir = ""     # PROMISE12 annotation direction
npy_data_dir = ""  # converted to npy image destination directory
npy_label_dir = "" #  annotation destionation directory

if not os.path.exists(npy_data_dir):
    os.makedirs(npy_data_dir)
if not os.path.exists(npy_label_dir):
    os.makedirs(npy_label_dir)

def save_npySlices (name, location, npy_dir):
    nrrd_file = os.path.join(location, name)
    v = sitk.ReadImage(nrrd_file)
    volume = sitk.GetArrayFromImage(v)
    all_slices = []
    for i in range(0,volume.shape[0]):
        slice_location = os.path.join(npy_dir, name[:len(name)-5] + '_' + str(i) + '.npy') 
        np.save(slice_location, volume[i, :, :])
        all_slices.append(slice_location)
    return all_slices

def saveSlicesAndCreateCSV(data_idx, csv_name):
    v = open(os.path.join(root_dir, csv_name), "w")             
    v.write("image,label\n")
    for idx, name in enumerate(data_idx):
        image_slices = save_npySlices(name, data_dir, npy_data_dir)
        labels_slices = save_npySlices(name, data_dir, npy_label_dir)   
        for slice_loc,label_loc in zip(image_slices, labels_slices):
            v.write("{},{}\n".format(slice_loc, label_loc))

def main():
    
    val_rate  = 0.1 # 10% of data for validation
    test_rate = 0.1 # 10% of data for test
    
    shuffle = 1
    data_list = []
    for file in os.listdir(data_dir):
        if file.endswith(".nrrd"):
            data_list.append(file)
    data_len    = len(data_list)
    val_len     = int(data_len * val_rate)
    test_len    = int(data_len * test_rate)

    if shuffle:
        data_idx = random.sample(range(data_len), data_len)
    else:
        data_idx = list(range(data_len))
    
    val_idx = [data_list[i] for i in data_idx[:val_len]]
    test_idx = [data_list[i] for i in data_idx[val_len:val_len+test_len]]
    train_idx = [data_list[i] for i in data_idx[val_len+test_len:]]

    saveSlicesAndCreateCSV(data_idx=val_idx, csv_name="val.csv") 
    saveSlicesAndCreateCSV(data_idx=test_idx, csv_name="test.csv") 
    saveSlicesAndCreateCSV(data_idx=train_idx, csv_name="train.csv")
    
if __name__ == "__main__"
    main()
