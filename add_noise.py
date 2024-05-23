#%%
import os
import glob
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from perlin_numpy import generate_perlin_noise_3d
import random

for num in range(0, 26):

    str_num = str(num)
    if len(str_num) == 1: str_num = "0" + str_num

    foldername = 'convert_to_npy/'
    filename = f'00000000{str_num}'
    extension = '.npy'
    filename_final = foldername + filename + extension # synthetic noisy ultrasound image 


    print('loading data ...')

    synth_us = np.load(filename_final)

    foldername2 = 'noise/'
    filename2 = f'noise{num}'
    extension2 = '.npy'
    filename2_final = foldername2 + filename2 + extension2

    noise = np.load(filename2_final)

    npy_with_noise = synth_us + noise

    np.save('npy_w_noise/' + filename + '.npy', npy_with_noise) 

    print(num)
