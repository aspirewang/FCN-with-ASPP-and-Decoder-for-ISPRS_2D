import numpy as np
import os
import scipy.io as io
import matplotlib.pyplot as plt
from PIL import Image
import skimage.io as skimage_io

#定义数据集的路径
base_dataset_dir_voc = '../../dataset/ISPRS_2D_512x512_s448/'

train_images_folder_name_voc = 'RGBIRDSM_train/'
train_annotations_folder_name_voc = '5_Labels_all_1D_train/'
train_images_dir_voc = os.path.join(base_dataset_dir_voc,train_images_folder_name_voc)
train_annotations_dir_voc = os.path.join(base_dataset_dir_voc, train_annotations_folder_name_voc)

val_images_folder_name_voc = 'RGBIRDSM_val_no_stride/'
val_annotations_folder_name_voc = '5_Labels_all_1D_val_no_stride/'
val_images_dir_voc = os.path.join(base_dataset_dir_voc,val_images_folder_name_voc)
val_annotations_dir_voc = os.path.join(base_dataset_dir_voc, val_annotations_folder_name_voc)

#读取文件列表
train_images_filename_list= os.listdir(train_images_dir_voc)
val_images_filename_list = os.listdir(val_images_dir_voc)
#print(val_images_filename_list[0:121])

test_list = ['top_potsdam_2_13', 'top_potsdam_2_14', 'top_potsdam_3_13', 'top_potsdam_3_14',
             'top_potsdam_4_13', 'top_potsdam_4_14', 'top_potsdam_4_15', 'top_potsdam_5_13',
             'top_potsdam_5_14', 'top_potsdam_5_15', 'top_potsdam_6_13', 'top_potsdam_6_14',
             'top_potsdam_6_15', 'top_potsdam_7_13']

def label_to_RGB(image):
    RGB = np.zeros(shape=[image.shape[0], image.shape[1], 3], dtype=np.uint8)
    index = image == 0
    RGB[index] = np.array([255, 255, 255])
    index = image == 1
    RGB[index] = np.array([0, 0, 255])
    index = image == 2
    RGB[index] = np.array([0, 255, 255])
    index = image == 3
    RGB[index] = np.array([0, 255, 0])
    index = image == 4
    RGB[index] = np.array([255, 255, 0])
    index = image == 5
    RGB[index] = np.array([255, 0, 0])
    return RGB


for image_name in test_list:
    w_patch = []
    h_patch = []
    for i in range(11):
        for j in range(11):
            name = image_name + '_' + str(i*11+j) + '.mat'
            location = val_images_filename_list.index(name)
            patch = io.loadmat('network_output/' + str(location + 1) + '.mat')['network_output']
            if j == 0:
                w_patch = patch
            else:
                w_patch = np.concatenate((w_patch, patch), axis=1)
        if i == 0:
            h_patch = w_patch
        else:
            h_patch = np.concatenate((h_patch, w_patch), axis=0)

    image = h_patch
    skimage_io.imsave('network_result_1D/' + image_name + '.png', image)
    image_RGB = label_to_RGB(image)
    skimage_io.imsave('network_result/' + image_name + '.tif', image_RGB)
    print(image_RGB.shape)
    plt.imshow(image_RGB)
    plt.show()



