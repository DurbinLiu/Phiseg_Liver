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
    Nimgs = 20
    channels = 1 #3
    height = 512  # 584
    width = 512  # 565
    num_labels_per_subject = 1  #几个人标记的

    data = {}
    list_data = []
    series_uid = 0
   #imgs = np.empty((Nimgs, height, width, channels)) #原来的
    imgs = np.empty((Nimgs, height, width))
    groundTruth = np.empty((Nimgs, num_labels_per_subject, height, width))

    for path, subdirs, files in os.walk(imgs_dir):  # list all files, directories in the path
        for i in range(len(files)):
            new_data = {}
            # original
            print("original image: ", files[i])
            series_uid = i
            img = Image.open(os.path.join(imgs_dir,files[i]))
            #img.show()
            img = img.convert('L') #自己加的，转成灰度图(512,512,3)->(512,512)
            #img.show()
            print("img.shape")
            imgs[i] = np.asarray(img )
            #imgs[i] = np.asarray(img).reshape((height, width, channels)) #自己加的，加了一个通道
            #img[i]  #自加，增加数组维度

            # corresponding ground truth
            groundTruth_name = files[i]
            print("ground truth name: ", groundTruth_name)
            g_truth = Image.open(os.path.join(groundTruth_dir, groundTruth_name))
            #g_truth.show()
            groundTruth[i][0] = np.asarray(g_truth,dtype = np.uint8)  #TODO 实际上是指定了第i张图片的第一维为0， 强行搞了四维，降了之后成为三维
            ground = groundTruth[i][0]
            ground = ground[np.newaxis,...]
            print('ground',ground.shape)

            #list_data = [('images',imgs[i]), ("masks",groundTruth[i]), ('series_uid',series_uid) ]
            list_data = [ ('image', imgs[i]), ("masks", ground) , ('series_uid', series_uid)]  #groundtruth[i]
            new_data[series_uid] = dict(list_data)

            data.update(new_data)

    print("imgs max: ", str(np.max(imgs)))
    print("imgs min: ", str(np.min(imgs)))
    assert (np.max(groundTruth) == 255 ) #and np.max(border_masks) == 255)
    assert (np.min(groundTruth) == 0 )   #and np.min(border_masks) == 0)
    print("ground truth and border masks are correctly withih pixel value range 0-255 (black-white)")

    # assert (imgs.shape == (Nimgs, height, width, channels))
    # groundTruth = np.reshape(groundTruth, (Nimgs, height, width, 1))
    # assert (groundTruth.shape == (Nimgs, height, width, 1))

    print('data:---')
    print(data)
    return data


def prepare_data(imgs_dir,groundTruth_dir, output_file):
    '''
    ！！！主要的函数，预处理数据集的！！！
    Main function that prepares a dataset from the raw challenge data to an hdf5 dataset
    '''

    hdf5_file = h5py.File(output_file, "w")

    data = {}

    data = get_datasets(imgs_dir, groundTruth_dir) #得到 data 字典

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


        #lbl = lbl.transpose((0,1))  #
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


def load_and_maybe_process_data(input_file,
                                preprocessing_folder,
                                force_overwrite=False):
    '''
    This function is used to load and if necessary preprocesses the LIDC challenge data

    :param input_folder: Folder where the raw ACDC（实际上是lidc数据，pickle文件） challenge data is located
    :param preprocessing_folder: Folder where the proprocessed data (hdf5文件) should be written to
    :param force_overwrite: Set this to True if you want to overwrite already preprocessed data [default: False]

    :return: Returns an h5py.File handle to the dataset  返回一个h5py文件处理句柄
    '''

    data_file_name = 'data_lidc.hdf5'

    data_file_path = os.path.join(preprocessing_folder, data_file_name)

    utils.makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path) or force_overwrite:
        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(input_file, data_file_path)
    else:
        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')  #返回一个h5py文件处理句柄


if __name__ == '__main__':

    imgs_dir = "C:\\Users\\123\Desktop\\bishe\Others\ganzang\data\\val\image"
    groundTruth_dir = "C:\\Users\\123\Desktop\\bishe\Others\ganzang\data\\val\label"

    preprocessing_folder = 'G:\\bishe\preproc_data\lidc'

    data_file_name = 'data_liver.hdf5'

    outfile = os.path.join(preprocessing_folder, data_file_name)
    #直接强制prepare_data了
    prepare_data(imgs_dir,groundTruth_dir,outfile)

