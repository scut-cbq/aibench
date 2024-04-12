import os
import argparse
import gdown
import requests
import hashlib

from typing import Union, Optional

def Args():
    parser = argparse.ArgumentParser()
    
    # proxy
    parser.add_argument('--proxy', type=str, default=None, help='use a proxy if needed (e.g., 127.0.0.1:7897)')

    # prepare data
    parser.add_argument('--skip_download', action='store_true')

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
    gdown.download(url, path, proxy=proxy)

def main():
    args = Args()
    if args.proxy:
        args.proxy = 'http://' + args.proxy

    # download traffic dataset
    URL = 'https://drive.google.com/uc?id=1U3BZ3Wvuvd9HVAx5Nl3bHYG9rsh5-yZX'
    MD5 = 'a62d8f2cd2c6acaaaa6f7aa819e378c0'
    DATASET_PATH = os.path.abspath('./dataset')
    DATA_PATH = os.path.abspath('./dataset/traffic.csv')
    os.makedirs(DATASET_PATH, exist_ok=True)

    if args.skip_download:
        print('Skip download')
    elif check_file(DATA_PATH, MD5):
        print('Datasets exist')
    else:
        print('Downloading datasets')
        try:
            download(URL, DATA_PATH, args.proxy)
            assert(check_file(DATA_PATH, MD5)==True)
        except requests.ConnectionError:
            print(f'ConnectionError. Try downloading manually from "{URL}". MD5 is "{MD5}"')


if __name__ == '__main__':
    main()