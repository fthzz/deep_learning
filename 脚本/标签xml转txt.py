# -- coding:utf-8 --
import xml.etree.ElementTree as ET
import os
from PIL import Image

# 类别，适用于已知类别
CLASSES = ['vertical', 'horizontal', 'single']

# xml,img文件路径,输出txt文件路径
xml_input = r"/home/fthzzz/Desktop/2110941_1716526530/Annotations"
imgs_input = r'/home/fthzzz/Desktop/2110941_1716526530/Images'
txt_output = r'/home/fthzzz/Desktop/2110941_1716526530/labels'  ##没有txt文件目录需要手动创建

# 自动识别类别，若已经有类别信息，注释该段和make_label_txt()的auto_classes(filenames)即可
CLASSES = []  # 类比列表


def auto_classes(filenames):
    global CLASSES
    for image_id in filenames:
        in_file = open(xml_input + '//' + image_id)
        tree = ET.parse(in_file)
        root = tree.getroot()
        for obj in root.iter("object"):
            obj_cls = obj.find("name").text
            CLASSES.append(obj_cls)
        CLASSES = list(set(CLASSES))  # set去重后是乱序，按照print出的CLASSES识别


def convert(size, box):
    # 将bbox的左上角点，右下角点坐标的格式，转换为bbox中心点+bbox的W,H的格式，并进行归一化
    # 读取对应图像尺
    dw = 1. / size[0]
    dh = 1. / size[1]
    # 转换为中心点坐标
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    # 归一化
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    # 把图像image_id的xml文件转换为目标检测的label文件(txt)
    # 其中包含物体的类别cls,bbox的中心点坐标,以及bbox的W,H
    # 并将四个物理量归一化
    in_file = open(xml_input + '//' + image_id)
    image_id = image_id.split(".")[0]  # 图片id
    tree = ET.parse(in_file)
    root = tree.getroot()

    # 获取图像尺寸
    img_path = imgs_input + '//' + image_id + r'.jpg'
    image = Image.open(img_path)
    width, height = image.size

    for obj in root.iter("object"):
        difficult = obj.find("difficult").text
        obj_cls = obj.find("name").text
        if obj_cls not in CLASSES or int(difficult) == 1:
            continue
        cls_id = CLASSES.index(obj_cls)
        xmlbox = obj.find("bndbox")
        points = (float(xmlbox.find("xmin").text),
                  float(xmlbox.find("xmax").text),
                  float(xmlbox.find("ymin").text),
                  float(xmlbox.find("ymax").text))
        bb = convert((width, height), points)

        # 打开文件以写入模式
        file_path = txt_output + image_id + r'.txt'  # 文件路径和名称
        with open(file_path, "a") as out_file:
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + "\n")


def make_label_txt():
    # labels文件夹下创建image_id.txt
    # 对应每个image_id.xml提取出的bbox信息
    filenames = os.listdir(xml_input)  # 获取文件名列表
    auto_classes(filenames)  # 注释该行
    print(CLASSES)
    for file in filenames:
        convert_annotation(file)


if __name__ == "__main__":
    # 开始提取和转换
    make_label_txt()
