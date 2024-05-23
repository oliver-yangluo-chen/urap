#%%
import os
import glob
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import math
import time

from misc import normalize, round_ptcloud


foldername = 'SarahWasinger_bases_heart_transformed_e15e410774/'
filename = '0000000025'
filename2 = '0000000026'
extension = '.npz'
filename_us = foldername + filename + extension # synthetic noisy ultrasound image 
filename_us2 = foldername + filename2 + extension


print('loading data ...')

synth_us = np.load(filename_us)['XYZ'] #there are multiple npy files with different names in this npz file
synth_us2 = np.load(filename_us2)['XYZ'] #there are multiple npy files with different names in this npz file

synth_us = normalize(synth_us) #convert all coordinates to 0-63
synth_us2 = normalize(synth_us2)


displace_ptcloud = synth_us2 - synth_us 

int_synth_us = round_ptcloud(synth_us) #convert all coordinates to integers

displace_voxel = np.zeros((64, 64, 64), dtype=[('x', 'float'), ('y', 'float'), ('z', 'float')]) #initialize npy array


for i in range(len(displace_ptcloud)): #set each voxel to its displacement (dx, dy, dz)
    coords = int_synth_us[i]
    d = tuple(displace_ptcloud[i])

    displace_voxel[int(coords[0]), int(coords[1]), int(coords[2])] = d

new_displace_voxel = np.zeros((64, 64, 64), dtype=[('x', 'float'), ('y', 'float'), ('z', 'float')]) #for bfs displacement values to empty voxels

DIRS = [
    (0, 0, 1), (0, 0, -1),
    (0, 1, 0), (0, -1, 0),
    (1, 0, 0), (-1, 0, 0)
]

def in_bounds(t):
    f = lambda x: x >= 0 and x < 64
    return f(t[0]) and f(t[1]) and f(t[2])

def dist(t1, t2):
    return math.sqrt((t2[0]-t1[0])**2 + (t2[1]-t1[1])**2 + (t2[2]-t1[2])**2)

def add_tuple(t1, t2):
    return tuple(t1[i] + t2[i] for i in range(len(t1)))

def mult_tuple(t, s):
    return tuple(x * s for x in t)

def bfs(i, j, k, ans): #no point in doing bfs
    neighbors = [(i, j, k)]
    old = displace_voxel[i, j, k]
    visited = set()
    visited.add((i, j, k))
    while len(neighbors) > 0:
        #print(len(visited), end = " ")
        #print(neighbors)
        cur_index = neighbors.pop(0)

        cur_new = ans[cur_index[0], cur_index[1], cur_index[2]] #current displacement in new array

        cur_new = add_tuple(cur_new, mult_tuple(old, (64-dist((i, j, k), cur_index))/64 )) # add old * (64-dist)/64, then divide by number of total displacements

        for d in DIRS:
            new_index = (cur_index[0] + d[0], cur_index[1] + d[1], cur_index[2] + d[2])
            if in_bounds(new_index) and new_index not in visited:
                neighbors.append(new_index)
                visited.add(new_index)
    print(len(visited))

def quadrant(t): #returns x, y, z range for iterate() to save time
    def help(x):
        return range(8 * int(x/8), 8 * int(x/8) + 8)
    return (help(t[0]), help(t[1]), help(t[2]))

def iterate(t, ans):
    old = displace_voxel[t[0], t[1], t[2]]
    ranges = quadrant(t)
    for i in ranges[0]:
        for j in ranges[1]:
            for k in ranges[2]:
                ans[i, j, k] = add_tuple(ans[i, j, k], mult_tuple(old, (64-dist((i, j, k), t))/(64*len(synth_us)/(8*8*8)) )) # add old * (64-dist)/64, then divide by number of total displacements in that quadrant
    return 

for i in range(64):
    for j in range(64):
        for k in range(64):
            #if np.equal(displace_voxel, np.zeros(3, dtype=[('x', float)])): continue
            if tuple(displace_voxel[i, j, k]) == (0., 0., 0.): continue
            if k % 10 == 0 and j % 10 == 0: print(i, j, k, tuple(displace_voxel[i, j, k]))
            #non-empty displacement value
            #bfs(i, j, k, new_displace_voxel)
            iterate((i, j, k), new_displace_voxel)

for i in range(64):
    for j in range(64):
        for k in range(64):
            if tuple(displace_voxel[i, j, k]) == (0., 0., 0.): continue
            new_displace_voxel[i, j, k] = displace_voxel[i, j, k]

np.save('displacements/' + filename + 'to' + filename2 + '.npy', new_displace_voxel) 

