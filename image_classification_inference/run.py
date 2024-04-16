import argparse
import torch
import torch.utils
import torch.utils.data
import torchvision
import os
import time
from tqdm import tqdm
from loguru import logger

def Args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='data/val', help='path to the dataset')
    parser.add_argument('--model_path', type=str, default='model/resnet50-19c8e357.pth', help='path to the pretrained model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size (default 32)')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(), choices=range(os.cpu_count() + 1), help='how many subprocesses to use for data loading (default all threads)')
    parser.add_argument('--gpu', type=int, default=0, help='use which gpu')

    return parser.parse_args()

def main():
    # args
    args = Args()

    device = f'cuda:{args.gpu}'

    # logger
    logger.level('bench', no=100, color='<magenta><bold>')
    logger.add(sink=f'results/batch{args.batch_size}.csv', format="{message}", encoding='utf-8', level='bench', mode='w')
    logger.log('bench', 'infer t (s),acc')

    # dataset
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_set = torchvision.datasets.ImageFolder(args.data_path, transform=transforms)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=args.batch_size, num_workers=args.num_workers)

    # model
    model = torchvision.models.resnet50()
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    model.eval()

    # inference
    correct = 0
    infer_time = 0

    with torch.no_grad():
        for x, y in tqdm(val_loader):
            start = time.time()
            x = x.to(device)

            pred = model(x)

            infer_time += time.time() - start
            correct += (pred.argmax(axis=1).cpu().apply_(lambda x: val_set.class_to_idx[str(x)])==y).sum().item()
    
    logger.log('bench', f'{infer_time},{correct/len(val_set)}')

if __name__ == '__main__':
    main()