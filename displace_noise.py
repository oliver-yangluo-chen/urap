import os
import glob
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt

for start in range(0, 26):


    noise = np.load(f'noise/noise{start}.npy')

    str1 = str(start)
    str2 = str(start + 1)

    if len(str1) == 1: str1 = "0" + str1
    if len(str2) == 1: str2 = "0" + str2

    displace = np.load(f'displacements/00000000{str1}to00000000{str2}.npy')

    #new_noise = np.zeros((64, 64, 64)) #initialize npy array
    new_noise = np.copy(noise)

    def in_bounds(t):
        f = lambda x: x >= 0 and x < 64
        return f(t[0]) and f(t[1]) and f(t[2])

    def keep_in_bounds(x):
        if x > 63: return (63 - (x - 63))%64
        if x < 0: return (-x)%64
        return x
        

    for i in range(64):
        for j in range(64):
            for k in range(64):
                cur_noise = noise[i, j, k]
                cur_displace = tuple(displace[i, j, k]) #(dx, dy, dz)

                for r in range(1):

                    newx = round(float(i) + cur_displace[0])
                    newy = round(float(j) + cur_displace[1])
                    newz = round(float(k) + cur_displace[2])

                    new_noise[keep_in_bounds(newx), keep_in_bounds(newy), keep_in_bounds(newz)] = cur_noise


    np.save(f'noise/noise{start+1}.npy', new_noise)

    print(start)
