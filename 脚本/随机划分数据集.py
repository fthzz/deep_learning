"""
    这是随机划分VOC数据集的脚本
"""

import os
import numpy as np


def random_split_voc_dataset(voc_dataset_dir, train_ratio=0.6,val_ratio=0.2):
    """
    这是随机划分VOC数据集为训练集和验证集的函数
    :param voc_dataset_dir: VOC数据集地址
    :param train_ratio: 训练集比例，默认为0.8
    :return:
    """
    # 初始化相关文件和文件夹路径
    # voc_main_dir = os.path.join(voc_dataset_dir,'ImageSets','Main')
    voc_main_dir = voc_dataset_dir
    voc_image_dir = os.path.join(voc_dataset_dir, 'Images')
    train_txt_path = os.path.join(voc_main_dir, 'train.txt')
    trainval_txt_path = os.path.join(voc_main_dir, 'label_list.txt')
    val_txt_path = os.path.join(voc_main_dir, 'val.txt')
    test_txt_path = os.path.join(voc_main_dir, 'test.txt')
    if not os.path.exists(voc_main_dir):
        os.makedirs(voc_main_dir)

    # 遍历图像文件夹，获取所有图像
    image_name_list = []
    for image_name in os.listdir(voc_image_dir):
        image_name_list.append(image_name)
    image_name_list = np.array(image_name_list)
    image_name_list = np.random.permutation(image_name_list)

    # 划分训练集和验证集
    size = len(image_name_list)
    random_index = np.random.permutation(size)#随机排列一个序列
    train_size = int(size * train_ratio)
    val_size = int(size * val_ratio)
    train_image_name_list = image_name_list[random_index[0:train_size]]
    val_image_name_list = image_name_list[random_index[train_size:train_size+val_size]]
    test_image_name_list = image_name_list[random_index[train_size + val_size:]]


    # 生成trainval
    with open(trainval_txt_path, 'w') as f:
        for image_name in image_name_list:
            fname, ext = os.path.splitext(image_name)
            fname = "./Images/" + fname + ".jpg" + " " + "./Annotations/" + fname + ".xml"
            f.write(fname + "\n")

    # 生成train
    with open(train_txt_path, 'w') as f:
        for image_name in train_image_name_list:
            fname, ext = os.path.splitext(image_name)
            fname = "./Images/" + fname + ".jpg" + " " + "./Annotations/" + fname + ".xml"
            f.write(fname + "\n")

    # 生成val
    with open(val_txt_path, 'w') as f:
        for image_name in val_image_name_list:
            fname, ext = os.path.splitext(image_name)
            fname = "./Images/" + fname + ".jpg" + " " + "./Annotations/" + fname + ".xml"
            f.write(fname + "\n")

    #生成test
    with open(test_txt_path, 'w') as f:
        for image_name in test_image_name_list:
            fname, ext = os.path.splitext(image_name)
            fname = "./Images/" + fname + ".jpg" + " " + "./Annotations/" + fname + ".xml"
            f.write(fname + "\n")

def run_main():
    """
    这是主函数
    """
    train_ratio = 0.6
    val_ratio=0.2
    # voc_dataset_dir = os.path.abspath("./"),
    voc_dataset_dir = '/home/fthzzz/Desktop/ecar-vision/data_detection/'
    random_split_voc_dataset(voc_dataset_dir, train_ratio,val_ratio)


if __name__ == '__main__':
    run_main()