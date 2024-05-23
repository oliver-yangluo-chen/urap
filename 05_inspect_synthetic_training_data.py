#%%
import os
import glob
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt


for num in range(0, 27):
    str_num = str(num)
    if len(str_num) == 1: str_num = "0" + str_num

    # str_num2 = str(num+1)
    # if len(str_num2) == 1: str_num2 = "0" + str_num2

    # foldername = 'displace_reprs/'
    # file = f'00000000{str_num}to00000000{str_num2}'
    # extension = '.npy'
    # filename_us = foldername + file + extension

    foldername = 'npy_w_noise/'
    file = f'00000000{str_num}'
    extension = '.npy'
    filename_us = foldername + file + extension




    print('loading data ...')

    synth_us = np.load(filename_us)

    if filename_us[-3:] == 'npz': synth_us = synth_us['XYZ']

    #synth_sg_wh = np.load(filename_sg_wh)
    #synth_sg_bv = np.load(filename_sg_bv)

    #print(synth_us.tolist())


    print('loaded from: ' + filename_us)
    print('file with ' + str(len(synth_us)) + ' 3D frames with type ' + str(synth_us.dtype))
    print('dimensions: ' + str(synth_us.shape[0]) +'x' + str(synth_us.shape[1]) + str(synth_us.shape[2]))


    n_frames = len(synth_us)

    #%%

    #i = random.randint(0,n_frames-1)
    #print('pick a random frame: ' + str(i))
    i=32 #nice example

    synth_us_3D = synth_us


    #synth_sgwh_3D = synth_sg_wh[i,:,:,:]
    #synth_sgbv_3D = synth_sg_bv[i,:,:,:]

    d = synth_us_3D.shape[0]
    dw=5
    c1 = int(d/2) # red
    c2 = int(d/2) # blue
    c3 = int(d/2)# green

    #sg1 = np.squeeze(synth_sgwh_3D[c1,:,:])
    #sg1 = np.transpose(sg1[::-1,::-1])
    #sg2 = np.squeeze(synth_sgwh_3D[:,c2,:])
    #sg2 = np.transpose(sg2[::-1,::-1])
    #sg3 = np.squeeze(synth_sgwh_3D[:,:,c3])
    #sg3 = np.transpose(sg3[::-1,::-1])


    us1 = np.squeeze(synth_us_3D[c1,:,:])
    us1 = np.transpose(us1[::-1,::-1])
    us2 = np.squeeze(synth_us_3D[:,c2,:])
    us2 = np.transpose(us2[::-1,::-1])
    us3 = np.squeeze(synth_us_3D[:,:,c3])
    us3 = np.transpose(us3[::-1,::-1])
    '''
    sgbv1 = np.squeeze(synth_sgbv_3D[c1,:,:]) # b-ventricular part of whole-heart segmentation
    sgbv1 = np.transpose(sgbv1[::-1,::-1])
    sgbv2 = np.squeeze(synth_sgbv_3D[:,c2,:])
    sgbv2 = np.transpose(sgbv2[::-1,::-1])
    sgbv3 = np.squeeze(synth_sgbv_3D[:,:,c3])
    sgbv3 = np.transpose(sgbv3[::-1,::-1])
    '''
    s=63

    fig, ax = plt.subplots(3, 3)
    #ax[0,0].imshow(sg1, cmap='gray', vmin=0, vmax=1)
    ax[0,0].plot([0, s], [s+1-c3, s+1-c3], color='green')
    ax[0,0].plot([c2, c2], [0, 63], color='blue')
    ax[0,0].plot([0, s, s, 0, 0], [0, 0, s, s, 0], color='red')
    ax[0,0].axis('off')
    #ax[0,1].imshow(sg2, cmap='gray', vmin=0, vmax=1)
    ax[0,1].plot([s+1-c1, s+1-c1], [0, s], color='red')
    ax[0,1].plot([0, s], [s+1-c3, s+1-c3], color='green')
    ax[0,1].plot([0, s, s, 0, 0], [0, 0, s, s, 0], color='blue')
    ax[0,1].axis('off')

    #ax[0,2].imshow(sg3, cmap='gray', vmin=0, vmax=1)
    ax[0,2].plot([0, s], [s+1-c2, s+1-c2], color='blue')
    ax[0,2].plot([s+1-c1, s+1-c1], [0, s], color='red')
    ax[0,2].plot([0, s, s, 0, 0], [0, 0, s, s, 0], color='green')
    ax[0,2].axis('off')

    ax[1,0].imshow(us1, cmap='gray', vmin=0, vmax=1)
    ax[1,0].plot([c2, c2], [0, s], color='blue')
    ax[1,0].plot([0, s], [s+1-c3, s+1-c3], color='green')
    ax[1,0].plot([0, s, s, 0, 0], [0, 0, s, s, 0], color='red')
    ax[1,0].axis('off')
    ax[1,1].imshow(us2, cmap='gray', vmin=0, vmax=1)
    ax[1,1].plot([s+1-c1, s+1-c1], [0, s], color='red')
    ax[1,1].plot([0, s], [s+1-c3, s+1-c3], color='green')
    ax[1,1].plot([0, s, s, 0, 0], [0, 0, s, s, 0], color='blue')
    ax[1,1].axis('off')
    ax[1,2].imshow(us3, cmap='gray', vmin=0, vmax=1)
    ax[1,2].plot([0, s], [s+1-c2, s+1-c2], color='blue')
    ax[1,2].plot([s+1-c1, s+1-c1], [0, s], color='red')
    ax[1,2].plot([0, s, s, 0, 0], [0, 0, s, s, 0], color='green')
    ax[1,2].axis('off')

    #ax[2,0].imshow(sgbv1, cmap='gray', vmin=0, vmax=1)
    ax[2,0].plot([c2, c2], [0, s], color='blue')
    ax[2,0].plot([0, s], [s+1-c3, s+1-c3], color='green')
    ax[2,0].plot([0, s, s, 0, 0], [0, 0, s, s, 0], color='red')
    ax[2,0].axis('off')
    #ax[2,1].imshow(sgbv2, cmap='gray', vmin=0, vmax=1)
    ax[2,1].plot([s+1-c1, s+1-c1], [0, s], color='red')
    ax[2,1].plot([0, s], [s+1-c3, s+1-c3], color='green')
    ax[2,1].plot([0, s, s, 0, 0], [0, 0, s, s, 0], color='blue')
    ax[2,1].axis('off')
    #ax[2,2].imshow(sgbv3, cmap='gray', vmin=0, vmax=1)
    ax[2,2].plot([0, s], [s+1-c2, s+1-c2], color='blue')
    ax[2,2].plot([s+1-c1, s+1-c1], [0, s], color='red')
    ax[2,2].plot([0, s, s, 0, 0], [0, 0, s, s, 0], color='green')
    ax[2,2].axis('off')




    plt.savefig(f"output_imgs/{file}.jpg")

    plt.show()



#%%

N=8


fig, ax = plt.subplots(N, 3)
for n in range(N):
    #i = random.randint(0,n_frames-1)
    i = N
    #synth_sg_3D = synth_sg[i,:,:,:]

    #sg1 = np.squeeze(synth_sg_3D[c1,:,:])
    #sg1 = np.transpose(sg1[::-1,::-1])
    #sg2 = np.squeeze(synth_sg_3D[:,c2,:])
    #sg2 = np.transpose(sg2[::-1,::-1])
    #sg3 = np.squeeze(synth_sg_3D[:,:,c3])
    #sg3 = np.transpose(sg3[::-1,::-1])

    d = synth_us_3D.shape[0]
    dw = 10 # how much random offset you want from the center plane
    c1 = int(d/2) + random.randint(-dw,dw) # pick a random cross-section at center pm 5
    c2 = int(d/2) + random.randint(-dw,dw)
    c3 = int(d/2) + random.randint(-dw,dw)

    ax[n,0].imshow(sg1, cmap='gray', vmin=0, vmax=1)
    ax[n,0].axis('off')
    ax[n,1].imshow(sg2, cmap='gray', vmin=0, vmax=1)
    ax[n,1].axis('off')
    ax[n,2].imshow(sg3, cmap='gray', vmin=0, vmax=1)
    ax[n,2].axis('off')
fig.set_tight_layout(True)

plt.show()


#%%


i = random.randint(0,n_frames-1)
print('pick a random frame: ' + str(i))

synth_us_3D = synth_us[i,:,:,:]
synth_sg_3D = synth_sg[i,:,:,:]

d = synth_us_3D.shape[0]
dw = 10 # how much random offset you want from the center plane
c1 = int(d/2) + random.randint(-dw,dw) # pick a random cross-section at center pm 5
c2 = int(d/2) + random.randint(-dw,dw)
c3 = int(d/2) + random.randint(-dw,dw)

sg1 = np.squeeze(synth_sg_3D[c1,:,:])
sg1 = np.transpose(sg1[::-1,::-1])
sg2 = np.squeeze(synth_sg_3D[:,c2,:])
sg2 = np.transpose(sg2[::-1,::-1])
sg3 = np.squeeze(synth_sg_3D[:,:,c3])
sg3 = np.transpose(sg3[::-1,::-1])

us1 = np.squeeze(synth_us_3D[c1,:,:])
us1 = np.transpose(us1[::-1,::-1])
us2 = np.squeeze(synth_us_3D[:,c2,:])
us2 = np.transpose(us2[::-1,::-1])
us3 = np.squeeze(synth_us_3D[:,:,c3])
us3 = np.transpose(us3[::-1,::-1])



fig, ax = plt.subplots(2, 3)
ax[0,0].imshow(sg1, cmap='gray', vmin=0, vmax=1)
ax[0,0].plot([0, 63], [64-c3, 64-c3], color='green')
ax[0,0].plot([c2, c2], [0, 63], color='blue')
ax[0,0].plot([0, 63, 63, 0, 0], [0, 0, 63, 63, 0], color='red')
ax[0,0].axis('off')
ax[0,1].imshow(sg2, cmap='gray', vmin=0, vmax=1)
ax[0,1].plot([64-c1, 64-c1], [0, 63], color='red')
ax[0,1].plot([0, 63], [64-c3, 64-c3], color='green')
ax[0,1].plot([0, 63, 63, 0, 0], [0, 0, 63, 63, 0], color='blue')
ax[0,1].axis('off')
ax[0,2].imshow(sg3, cmap='gray', vmin=0, vmax=1)
ax[0,2].plot([0, 63], [64-c2, 64-c2], color='blue')
ax[0,2].plot([64-c1, 64-c1], [0, 63], color='red')
ax[0,2].plot([0, 63, 63, 0, 0], [0, 0, 63, 63, 0], color='green')
ax[0,2].axis('off')
ax[1,0].imshow(us1, cmap='gray', vmin=0, vmax=1)
ax[1,0].plot([c2, c2], [0, 63], color='blue')
ax[1,0].plot([0, 63], [64-c3, 64-c3], color='green')
ax[1,0].plot([0, 63, 63, 0, 0], [0, 0, 63, 63, 0], color='red')
ax[1,0].axis('off')
ax[1,1].imshow(us2, cmap='gray', vmin=0, vmax=1)
ax[1,1].plot([64-c1, 64-c1], [0, 63], color='red')
ax[1,1].plot([0, 63], [64-c3, 64-c3], color='green')
ax[1,1].plot([0, 63, 63, 0, 0], [0, 0, 63, 63, 0], color='blue')
ax[1,1].axis('off')
ax[1,2].imshow(us3, cmap='gray', vmin=0, vmax=1)
ax[1,2].plot([0, 63], [64-c2, 64-c2], color='blue')
ax[1,2].plot([64-c1, 64-c1], [0, 63], color='red')
ax[1,2].plot([0, 63, 63, 0, 0], [0, 0, 63, 63, 0], color='green')
ax[1,2].axis('off')












#%%


N=6
M=4
P=N*M

fig, ax = plt.subplots(N, M)
for n in range(N):
    for m in range(M):
        i = random.randint(0,n_frames-1)
        synth_us_3D = synth_us[i,:,:,:]
        d = synth_us_3D.shape[0]
        dw = 10 # how much random offset you want from the center plane
        c1 = int(d/2) + random.randint(-dw,dw) # pick a random cross-section at center pm 5
        c2 = int(d/2) + random.randint(-dw,dw)
        c3 = int(d/2) + random.randint(-dw,dw)

        ax[n,m].imshow(np.squeeze(synth_us_3D[c1,:,:]), cmap='gray', vmin=0, vmax=1)
        ax[n,m].axis('off')
fig.set_tight_layout(True)

# %%