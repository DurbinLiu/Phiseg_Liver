# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import os
import numpy as np
import logging
import h5py
import pickle
from sklearn.model_selection import train_test_split
from PIL import Image

import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

#填充或剪切slice至指定大小
def crop_or_pad_slice_to_size(slice, nx, ny):
    x, y = slice.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.zeros((nx, ny))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :] = slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice[:, :]

    return slice_cropped

#找到id所属的子集
def find_subset_for_id(ids_dict, id):

    for tt in ['test', 'train', 'val']:
        if id in ids_dict[tt]:
            return tt
    raise ValueError('id was not found in any of the train/test/val subsets.')

def get_datasets(imgs_dir, groundTruth_dir):

    # 图像原始数量、通道数、高、宽
    Nimgs = 400
    channels = 1 #3
    height = 512  # 584
    width = 512  # 565
    num_labels_per_subject = 4  #几个人标记的
    target_size = (128,128)# 使图像  (512,512)->(128,128)方便训练
    #target_size = (512,512)# todo 原图

    data = {}
    # imgs = np.empty((Nimgs, height, width, channels)) #原来的
    imgs = np.empty((Nimgs, height, width))

    for path, subdirs, files in os.walk(imgs_dir):  # list all files, directories in the path
        for i in range(len(files)):
            new_data = {}

            # 创建groundTruth列表并初始化
            groundTruth = []

            print("original image: ", files[i])
            series_uid = i
            img = Image.open(os.path.join(imgs_dir, files[i]))
            img = img.convert('L')  # 自己加的，转成灰度图(512,512,3)->(512,512)
            #img.show()

            # 展示resize后的原图
            img = img.resize(target_size)
            #img.show()

            imgs2 = np.asarray(img,dtype = float)/255   # 参考他的 归一化
            print("img2 shape ：", imgs2.shape)


            # corresponding ground truth
            groundTruth_name = files[i]
            print("ground truth name: ", groundTruth_name)
            g_truth = Image.open(os.path.join(groundTruth_dir, groundTruth_name))
            g_truth = g_truth.resize(target_size)
            #g_truth.show()
            for j in range(num_labels_per_subject):
                #groundTruth.append(np.asarray(g_truth, dtype=np.uint8))  # TODO 0表示第一个列表元素是这个掩膜数组，实际上有多个label之后，这里要改]
                groundTruth.append(np.asarray(g_truth, dtype=np.float)/255)

            print('groundTruth shape : ', groundTruth[0].shape)
            print('groundTruth 长度：', len(groundTruth))

            list_data = [('image', imgs2), ("masks", groundTruth), ('series_uid', series_uid),("pixel_spacing",['0.714','0.714'])]
            new_data[series_uid] = dict(list_data)

            data.update(new_data)

    print("imgs max: ", str(np.max(imgs)))
    print("imgs min: ", str(np.min(imgs)))


    return data


def prepare_data(imgs_dir,groundTruth_dir, output_file):
    '''
    ！！！主要的函数，预处理数据集的！！！
    Main function that prepares a dataset from the raw challenge data to an hdf5 dataset
    '''

    hdf5_file = h5py.File(output_file, "w")

    data = {}

    data = get_datasets(imgs_dir, groundTruth_dir) #得到 data 字典

    print('data:---')
    print(data)

    series_uid = []

    for key, value in data.items(): #items() 方法把字典中每对 key 和 value 组成一个元组，并把这些元组放在列表中返回。
        print("!!! key and value")
        print(key,value)
        print("!!!")
        series_uid.append(value['series_uid']) #把series_uid都扩充到 series_uid列表里面

    unique_subjects = np.unique(series_uid) #返回The sorted unique values.

    #得到三个部分数据集的id
    split_ids = {}
    train_and_val_ids, split_ids['test'] = train_test_split(unique_subjects, test_size=0.2)
    split_ids['train'], split_ids['val'] = train_test_split(train_and_val_ids, test_size=0.2)

    images = {}
    labels = {}
    uids = {}
    groups = {}

    for tt in ['train', 'test', 'val']:
        # 列表初始化
        images[tt] = []
        labels[tt] = []
        uids[tt] = []
        groups[tt] = hdf5_file.create_group(tt) #写入hdf5操作在这里呢


    for key, value in data.items():

        s_id = value['series_uid'] #value还是一个字典，key没什么用这里

        tt = find_subset_for_id(split_ids, s_id) #寻找到id所在的子集，比如train

        images[tt].append(value['image'].astype(float)-0.5)

        lbl = np.asarray(value['masks'])  # this will be of shape 4 x 128 x 128  4表示4个experts？


        #lbl = lbl[np.newaxis, : , :]  # todo 这句是我加的下句是本来的
        lbl = lbl.transpose((1,2,0)) #（channels,imagesize,imagesize）转化为（imagesize,imagesize,channels）
        #关于这个转置函数：https://blog.csdn.net/u012762410/article/details/78912667


        labels[tt].append(lbl)
        uids[tt].append(hash(s_id))  # Checked manually that there are no collisions

    #真正写入到hdf5的也就是这三个groups，那是不是只要我找到方法这样写入就行了？
    for tt in ['test', 'train', 'val']:
        print("here:tt!!!")

        groups[tt].create_dataset('uids', data=np.asarray(uids[tt], dtype=np.int64)) #改动这个debug (python int too long)
        # 若拿pickle文件进行prepare，会报错啊
        # 说python的int型无限，转换不了c的int型。。。————要改成int32/64

        #所以到底要下载原来的image图片吗，太大了啊，怎么下载，下载了放在哪个文件夹？ ————不用下载初始的
        #再说了，pickle文件不就是已经预处理过了的吗，为什么还要处理，真奇怪、、————pickle是原文件，要预处理

        groups[tt].create_dataset('labels', data=np.asarray(labels[tt], dtype=np.uint8))
        groups[tt].create_dataset('images', data=np.asarray(images[tt], dtype=np.float))

    hdf5_file.close()



if __name__ == '__main__':

    imgs_dir = "E:\\bishe_durbin\Others\ganzang\data\\train\image"
    groundTruth_dir = "E:\\bishe_durbin\Others\ganzang\data\\train\label"

    preprocessing_folder = 'E:\\bishe_durbin\\46-PHiSeg-code-master-paper355\data\preproc_data\lidc'

    # todo
    data_file_name = 'data_liver_128_128.hdf5'
    #data_file_name = 'data_liver_512_512.hdf5'

    outfile = os.path.join(preprocessing_folder, data_file_name)
    #直接强制prepare_data了
    prepare_data(imgs_dir,groundTruth_dir,outfile)

