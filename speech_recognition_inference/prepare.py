import os
import argparse
import requests
import hashlib
import shutil
import json
import glob
import multiprocessing
import functools
import pandas as pd
import subprocess

from typing import Union, Optional
from tqdm import tqdm

def Args():
    parser = argparse.ArgumentParser()
    
    # proxy
    parser.add_argument('--proxy', type=str, default=None, help='use a proxy if needed (e.g., 127.0.0.1:7897)')

    # prepare data
    parser.add_argument('--skip_download', action='store_true')
    parser.add_argument('--skip_extract', action='store_true')
    parser.add_argument('--skip_install', action='store_true')
    parser.add_argument('--skip_convert', action='store_true')
    parser.add_argument('--num_process', type=int, default=os.cpu_count(), choices=range(os.cpu_count()+1), help='how many subprocesses used to process data (default all threads)')

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

def build_input_arr(input_dir):
    txt_files = glob.glob(os.path.join(input_dir, '**', '*.trans.txt'),
                          recursive=True)
    input_data = []
    for txt_file in txt_files:
        rel_path = os.path.relpath(txt_file, input_dir)
        with open(txt_file) as fp:
            for line in fp:
                fname, _, transcript = line.partition(' ')
                input_data.append(dict(input_relpath=os.path.dirname(rel_path),
                                       input_fname=fname + '.flac',
                                       transcript=transcript))
    return input_data

def preprocess(data, input_dir, dest_dir):
    os.environ['PATH'] += ':' + os.path.abspath('third_party/install/bin')
    import sox

    speed = 1
    input_fname = os.path.join(input_dir,
                               data['input_relpath'],
                               data['input_fname'])

    os.makedirs(os.path.join(dest_dir, data['input_relpath']), exist_ok=True)

    output_dict = {}
    output_dict['transcript'] = data['transcript'].lower().strip()
    output_dict['files'] = []

    fname = os.path.splitext(data['input_fname'])[0]
    output_fname = fname + '.wav'
    output_fpath = os.path.join(dest_dir,
                                data['input_relpath'],
                                output_fname)

    if not os.path.exists(output_fpath):
        cbn = sox.Transformer()
        cbn.build(input_fname, output_fpath)

    file_info = sox.file_info.info(output_fpath)
    file_info['fname'] = os.path.join(os.path.basename(dest_dir),
                                        data['input_relpath'],
                                        output_fname)
    file_info['speed'] = speed
    output_dict['files'].append(file_info)

    file_info = sox.file_info.info(output_fpath)
    output_dict['original_duration'] = file_info['duration']
    output_dict['original_num_samples'] = file_info['num_samples']

    return output_dict

def convert(dataset_path, num_process):
    input_dir = os.path.join(dataset_path, 'LibriSpeech', 'dev-clean')
    dest_dir = os.path.join(dataset_path, 'dev-clean-wav')
    dataset = build_input_arr(input_dir)

    with multiprocessing.Pool(num_process) as p:
        func = functools.partial(preprocess, input_dir=input_dir, dest_dir=dest_dir)
        dataset = list(tqdm(p.imap(func, dataset), total=len(dataset)))
    
    df = pd.DataFrame(dataset, dtype=object)

    # Save json with python. df.to_json() produces back slashed in file paths
    output_json = os.path.join(dataset_path, 'dev-clean-wav.json')
    dataset = df.to_dict(orient='records')
    with open(output_json, 'w') as fp:
        json.dump(dataset, fp, indent=2)


def main():
    args = Args()
    args.proxy = {'https': args.proxy}

    # download OpenSLR LibriSpeech ASR corpus dev-clean
    DATA_URL = 'https://www.openslr.org/resources/12/dev-clean.tar.gz'
    DATA_MD5 = '42e2234ba48799c1f50f24a7926300a1'
    DATASET_PATH = os.path.abspath('./data/')
    DATA_PATH = os.path.abspath('./data/dev-clean.tar.gz')
    os.makedirs(DATASET_PATH, exist_ok=True)

    if args.skip_download:
        print('Skip download')
    elif check_file(DATA_PATH, DATA_MD5):
        print('Datasets exist')
    else:
        print('Downloading datasets')
        try:
            download(DATA_URL, DATA_PATH, args.proxy)
            assert(check_file(DATA_PATH, DATA_MD5)==True)
        except requests.ConnectionError:
            print(f'ConnectionError. Try downloading manually from "{DATA_URL}". MD5 is "{DATA_MD5}"')
    
    # extract OpenSLR LibriSpeech ASR corpus dev-clean
    if args.skip_extract:
        print('Skip extract')
    else:
        print('Extracting datasets')
        extracted_dir = os.path.join(DATASET_PATH, 'LibriSpeech')
        if os.path.exists(extracted_dir):
            shutil.rmtree(extracted_dir)
        extract(DATA_PATH, DATASET_PATH)

    # install sox to convert .flac to .wav
    if args.skip_install:
        print('Skip install')
    else:
        print('Installing sox')
        third_party_dir = os.path.abspath('third_party')
        install_dir = f'{third_party_dir}/install'
        os.makedirs(install_dir, exist_ok=True)

        FLAC_URL= 'https://ftp.osuosl.org/pub/xiph/releases/flac/flac-1.3.2.tar.xz'
        FLAC_PATH = f'{third_party_dir}/flac-1.3.2.tar.xz'
        try:
            download(FLAC_URL, FLAC_PATH, args.proxy)
        except requests.ConnectionError:
            print(f'ConnectionError. Try downloading manually from "{FLAC_URL}"')
        extract(FLAC_PATH, third_party_dir)
        cmd = f'cd {third_party_dir}/flac-1.3.2; ./configure --prefix={install_dir} && make && make install'
        subprocess.check_call(cmd, shell=True)

        SOX_URL= 'https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2.tar.gz'
        SOX_PATH = f'{third_party_dir}/sox-14.4.2.tar.gz'
        try:
            download(SOX_URL, SOX_PATH, args.proxy)
        except requests.ConnectionError:
            print(f'ConnectionError. Try downloading manually from "{SOX_URL}"')
        extract(SOX_PATH, third_party_dir)
        cmd = f'cd {third_party_dir}/sox-14.4.2; LDFLAGS="-L{install_dir}/lib" CFLAGS="-I{install_dir}/include" ./configure --prefix={install_dir} --with-flac && make && make install'
        subprocess.check_call(cmd, shell=True)

    # convert .flac to .wav
    if args.skip_convert:
        print('Skip convert')
    else:
        print('Converting .flac to .wav')
        convert(DATASET_PATH, args.num_process)

    # download pretrained model
    MODEL_URL = 'https://zenodo.org/record/3662521/files/DistributedDataParallel_1576581068.9962234-epoch-100.pt?download=1'
    MODEL_MD5 = '286e7813dffbc65e7d86f721b7739f1c'
    MODEL_PATH = os.path.abspath('./model/rnnt.pt')
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