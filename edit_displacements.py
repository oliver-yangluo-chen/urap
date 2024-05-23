#%%
import os
import glob
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import math

for num in range(0, 26):

    str_num = str(num)
    if len(str_num) == 1: str_num = "0" + str_num

    str_num2 = str(num+1)
    if len(str_num2) == 1: str_num2 = "0" + str_num2

    foldername = 'displacements/'
    filename = f'00000000{str_num}to00000000{str_num2}'
    extension = '.npy'
    filename_final = foldername + filename + extension

    d = np.load(filename_final)

    npy_arr = np.load(f'convert_to_npy/00000000{str_num}.npy')

    def mult_tuple(t, s):
        return (x * s for x in t)
    
    def dist(a, b, c):
        return math.sqrt(a**2 + b**2 + c**2)
    
    #displace_repr = np.zeros((64, 64, 64)) #initialize npy array

    for i in range(64):
        for j in range(64):
            for k in range(64):

                x = d[i, j, k]['x']
                y = d[i, j, k]['y']
                z = d[i, j, k]['z']

                if npy_arr[i, j, k] == 0:
                    x = x / 35
                    y = y / 35
                    z = z / 35


    np.save(f'displacements/{filename + extension}', d)
    print(num)
