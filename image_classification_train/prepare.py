import os
import argparse
import requests
import hashlib
import shutil
import json
import xml.dom.minidom as XML
import numpy as np

from typing import Union, Optional
from tqdm import tqdm

def Args():
    parser = argparse.ArgumentParser()

    # proxy
    parser.add_argument('--proxy', type=str, default=None, help='use a proxy if needed (e.g., 127.0.0.1:7897)')

    # prepare data
    parser.add_argument('--skip_download', action='store_true')
    parser.add_argument('--skip_extract', action='store_true')
    parser.add_argument('--skip_prepare', action='store_true')

    return parser.parse_args()

# check if the file exists and the md5 is correct
def check_file(path: Union[str, list], md5: Optional[Union[str, list]]=None):
    if isinstance(path, str):
        path = [path]
    if isinstance(md5, str):
        md5 = [md5]
    if md5 is None:
        md5 = [None]*len(path)

    for p, m in zip(path, md5):
        if not os.path.exists(p):
            return False
        
        if m:
            with open(p, 'rb') as f:
                data = f.read()
            if hashlib.md5(data).hexdigest() != m:
                return False
    
    return True

def download(url, path, proxy=None):
    response = requests.get(url, stream=True, timeout=10, proxies=proxy)
    total_size = int(response.headers.get('content-length', 0))

    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=path)
    with open(path, 'wb') as f:
        for data in response.iter_content(chunk_size=1024):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

def extract(file, to):
    print(f'Extracting {file}')
    shutil.unpack_archive(file, to)

voc_cls_id = {"aeroplane":0, "bicycle":1, "bird":2, "boat":3, "bottle":4,
               "bus":5, "car":6, "cat":7, "chair":8, "cow":9,
               "diningtable":10, "dog":11, "horse":12, "motorbike":13, "person":14,
               "pottedplant":15, "sheep":16, "sofa":17, "train":18, "tvmonitor":19}


def get_label(data_path):
    print("generating labels for VOC07 dataset")
    xml_paths = os.path.join(data_path, "VOC2007/Annotations/")
    save_dir = "data/voc07/labels"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in os.listdir(xml_paths):
        if not i.endswith(".xml"):
            continue
        s_name = i.split('.')[0] + ".txt"
        s_dir = os.path.join(save_dir, s_name)
        xml_path = os.path.join(xml_paths, i)
        DomTree = XML.parse(xml_path)
        Root = DomTree.documentElement

        obj_all = Root.getElementsByTagName("object")
        leng = len(obj_all)
        cls = []
        difi_tag = []
        for obj in obj_all:
            # get the classes
            obj_name = obj.getElementsByTagName('name')[0]
            one_class = obj_name.childNodes[0].data
            cls.append(voc_cls_id[one_class])

            difficult = obj.getElementsByTagName('difficult')[0]
            difi_tag.append(difficult.childNodes[0].data)

        for i, c in enumerate(cls):
            with open(s_dir, "a") as f:
                f.writelines("%s,%s\n" % (c, difi_tag[i]))


def transdifi(data_path):
    print("generating final json file for VOC07 dataset")
    label_dir = "data/voc07/labels/"
    img_dir = os.path.join(data_path, "VOC2007/JPEGImages/")

    # get trainval test id
    id_dirs = os.path.join(data_path, "VOC2007/ImageSets/Main/")
    f_train = open(os.path.join(id_dirs, "train.txt"), "r").readlines()
    f_val = open(os.path.join(id_dirs, "val.txt"), "r").readlines()
    f_trainval = f_train + f_val
    f_test = open(os.path.join(id_dirs, "test.txt"), "r")

    trainval_id =  np.sort([int(line.strip()) for line in f_trainval]).tolist()
    test_id = [int(line.strip()) for line in f_test]
    trainval_data = []
    test_data = []

    # ternary label
    # -1 means negative
    # 0 means difficult
    # +1 means positive

    # binary label
    # 0 means negative
    # +1 means positive 

    # we use binary labels in our implementation

    for item in sorted(os.listdir(label_dir)):
        with open(os.path.join(label_dir, item), "r") as f:

            target = np.array([-1] * 20)
            classes = []
            diffi_tag = []

            for line in f.readlines():
                cls, tag = map(int, line.strip().split(','))
                classes.append(cls)
                diffi_tag.append(tag)

            classes = np.array(classes)
            diffi_tag = np.array(diffi_tag)
            for i in range(20):
                if i in classes:
                    i_index = np.where(classes == i)[0]
                    if len(i_index) == 1:
                        target[i] = 1 - diffi_tag[i_index]
                    else:
                        if len(i_index) == sum(diffi_tag[i_index]):
                            target[i] = 0
                        else:
                            target[i] = 1
                else:
                    continue
            img_path = os.path.join(img_dir, item.split('.')[0]+".jpg")

            if int(item.split('.')[0]) in trainval_id:
                target[target == -1] = 0  # from ternary to binary by treating difficult as negatives
                data = {"target": target.tolist(), "img_path": img_path}      
                trainval_data.append(data)
            if int(item.split('.')[0]) in test_id:
                data = {"target": target.tolist(), "img_path": img_path}      
                test_data.append(data)

    json.dump(trainval_data, open("data/voc07/trainval_voc07.json", "w"))
    json.dump(test_data, open("data/voc07/test_voc07.json", "w"))
    print("VOC07 data preparing finished!")
    print("data/voc07/trainval_voc07.json data/voc07/test_voc07.json")
    
    # remove label cash
    for item in os.listdir(label_dir):
        os.remove(os.path.join(label_dir, item))
    os.rmdir(label_dir)

def main():
    args = Args()
    args.proxy = {'http': args.proxy}

    # download VOC2007
    TRAIN_VAL_URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar'
    TEST_URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar'
    TRAIN_VAL_MD5 = 'c52e279531787c972589f7e41ab4ae64'
    TEST_MD5 = 'b6e924de25625d8de591ea690078ad9f'
    DATASET_PATH = os.path.abspath('./Dataset')
    TRAIN_VAL_PATH = os.path.abspath('./Dataset/VOCtrainval_06-Nov-2007.tar')
    TEST_PATH = os.path.abspath('./Dataset/VOCtest_06-Nov-2007.tar')
    os.makedirs(DATASET_PATH, exist_ok=True)

    if args.skip_download:
        print('Skip download')
    elif check_file([TRAIN_VAL_PATH, TEST_PATH], [TRAIN_VAL_MD5, TEST_MD5]):
        print('Datasets exist')
    else:
        print('Downloading datasets')
        try:
            download(TRAIN_VAL_URL, TRAIN_VAL_PATH, args.proxy)
            download(TEST_URL, TEST_PATH, args.proxy)
            assert(check_file([TRAIN_VAL_PATH, TEST_PATH], [TRAIN_VAL_MD5, TEST_MD5])==True)
        except requests.ConnectionError:
            print(f'ConnectionError. Try downloading manually from "{TRAIN_VAL_URL}" and "{TEST_URL}". MD5s are "{TRAIN_VAL_MD5}" and "{TEST_MD5}"')
    
    # extract VOC2007
    if args.skip_extract:
        print('Skip extract')
    else:
        print('Extracting datasets')
        extract(TRAIN_VAL_PATH, DATASET_PATH)
        extract(TEST_PATH, DATASET_PATH)

    # prepare data
    if args.skip_prepare:
        print('Skip prepare')
    else:
        print('Preparing data')
        if not os.path.exists("data/voc07"):
            os.makedirs("data/voc07")
        
        DATA_PATH = os.path.join(DATASET_PATH, 'VOCdevkit/')
        get_label(DATA_PATH)
        transdifi(DATA_PATH)

if __name__ == '__main__':
    main()