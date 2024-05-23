#%%
import os
import glob
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from perlin_numpy import generate_perlin_noise_3d
import random

for num in range(1, 27):
    str_num = str(num)
    if len(str_num) == 1: str_num = "0" + str_num

    foldername = 'SarahWasinger_bases_heart_transformed_e15e410774/'
    filename = f'00000000{str_num}'
    extension = '.npz'
    filename_us = foldername + filename + extension # synthetic noisy ultrasound image 


    print('loading data ...')

    synth_us = np.load(filename_us)['XYZ'] #there are multiple npy files with different names in this npz file

    minx = 0
    miny = 0
    minz = 0

    maxx = 0
    maxy = 0
    maxz = 0

    for p in synth_us: #[x, y, z] represents a point in space
        minx = min(minx, p[0])
        miny = min(miny, p[1])
        minz = min(minz, p[2])

        maxx = max(maxx, p[0])
        maxy = max(maxy, p[1])
        maxz = max(maxz, p[2])

    print("x", minx, maxx, "range", maxx - minx)
    print("y", miny, maxy, "range", maxy - miny)
    print("z", minz, maxz, "range", maxz - minz) #get dimensions

    npy_arr = np.zeros((64, 64, 64)) #initialize npy array

    for p in synth_us:
        for i in range(16):
            newx = round((p[0] - minx) * 63/(maxx - minx) + random.uniform(-0.25, 0.25))
            newy = round((p[1] - miny) * 63/(maxy - miny) + random.uniform(-0.25, 0.25))
            newz = round((p[2] - minz) * 63/(maxz - minz) + random.uniform(-0.25, 0.25))

            npy_arr[newx][newy][newz] = random.uniform(0.175, 0.825) #convert list of points to voxel format


        # newx = round((p[0] + 43.5) * 63/68)
        # newy = round((p[1] + 70) * 63/72)
        # newz = round((p[2] + 34) * 63/71)

        # npy_arr[newx][newy][newz] = 1 #convert list of points to voxel format

    #add noise
    seed = 123
    np.random.seed(seed)

    def make_perlin(res):
        noise = generate_perlin_noise_3d( (64, 64, 64), (res, res, res), tileable=(True, True, True))
        return noise * 0.5

    def make_random(): #Adds random noise to the mask, centered at 0 with standard deviation sigma
        noise = np.random.normal(0, 0.25, (64, 64, 64))
        return noise * 0.75

    noise = np.zeros((64, 64, 64)) #initialize npy array

    noise = noise + make_perlin(4)
    noise = noise + make_perlin(8)
    noise = noise + make_perlin(16)
    noise = noise + make_perlin(16) * 1.5

    noise = noise + make_perlin(2) * 2.25

    noise = noise + make_random()

    np.save('noise.npy', noise)

    # npy_arr = npy_arr + noise

    #save npy_arr

    #np.save('convert_to_npy/' + filename + '.npy', npy_arr) 
    print(num)

