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
    parser.add_argument('--skip_classify', action='store_true')

    # prepare model
    parser.add_argument('--skip_model', action='store_true')

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

def classify(DATASET_PATH, IMGS_PATH, VAL_MAP_PATH):
    with open(VAL_MAP_PATH, 'r') as f:
        lines = [line.rstrip() for line in f.readlines()]
    
    for i in range(1000):
        class_dir = os.path.join(IMGS_PATH, str(i))
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)

    for line in lines:
        path, label = line.split('\t')
        file_path = os.path.join(DATASET_PATH, path)
        class_dir = os.path.join(IMGS_PATH, str(label))
        if os.path.exists(file_path):
            shutil.move(file_path, class_dir)



def main():
    args = Args()
    args.proxy = {'https': args.proxy}

    # download ILSVRC2012 val
    VAL_URL = 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar'
    VAL_MAP_URL = 'https://github.com/microsoft/Swin-Transformer/files/8529898/val_map.txt'
    VAL_MD5 = '29b22e2961454d5413ddabcf34fc5622'
    VAL_MAP_MD5 = '6d562c1508c71064e224be7ad4a69b6f'
    DATASET_PATH = os.path.abspath('./data/')
    VAL_PATH = os.path.abspath('./data/ILSVRC2012_img_val.tar')
    VAL_MAP_PATH = os.path.abspath('./data/val_map.txt')
    os.makedirs(DATASET_PATH, exist_ok=True)

    if args.skip_download:
        print('Skip download')
    elif check_file([VAL_PATH, VAL_MAP_PATH], [VAL_MD5, VAL_MAP_MD5]):
        print('Datasets exist')
    else:
        print('Downloading datasets')
        try:
            download(VAL_URL, VAL_PATH, args.proxy)
            download(VAL_MAP_URL, VAL_MAP_PATH, args.proxy)
            assert(check_file([VAL_PATH, VAL_MAP_PATH], [VAL_MD5, VAL_MAP_MD5])==True)
        except requests.ConnectionError:
            print(f'ConnectionError. Try downloading manually from "{VAL_URL}". MD5 is "{VAL_MD5}"')
    
    # extract ILSVRC2012 val
    IMGS_PATH = os.path.join(DATASET_PATH, 'val')

    if args.skip_extract:
        print('Skip extract')
    else:
        print('Extracting datasets')
        extract(VAL_PATH, IMGS_PATH)

    # classify val images according to val_map.txt
    if args.skip_classify:
        print('Skip classify')
    else:
        print('Classifying images')
        classify(DATASET_PATH, IMGS_PATH, VAL_MAP_PATH)

    # download pretrained ResNet-50
    MODEL_URL = 'https://zenodo.org/record/4588417/files/resnet50-19c8e357.pth'
    MODEL_MD5 = '9e9c86b324d80e65229fab49b8d9a8e8'
    MODEL_PATH = os.path.abspath('./model/resnet50-19c8e357.pth')
    os.makedirs('./model', exist_ok=True)

    if args.skip_model:
        print('Skip model')
    elif check_file(MODEL_PATH, MODEL_MD5):
        print('Model exists')
    else:
        print('Downloading model')
        try:
            download(MODEL_URL, MODEL_PATH, args.proxy)
            assert(check_file(MODEL_PATH, MODEL_MD5)==True)
        except requests.ConnectionError:
            print(f'ConnectionError. Try downloading manually from "{MODEL_URL}". MD5 is "{MODEL_MD5}"')

if __name__ == '__main__':
    main()