import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger

from utils.evaluation.eval import evaluation
from utils.evaluation.warmUpLR import WarmUpLR
from utils.pipeline.resnet_csra import ResNet_CSRA
from utils.pipeline.dataset import DataSet

def Args():
    parser = argparse.ArgumentParser(description="settings")

    # public args
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(), choices=range(os.cpu_count() + 1), help='how many subprocesses to use for data loading (default all threads)')
    parser.add_argument('--gpu', type=int, default=0, help='use which gpu')

    args = parser.parse_args()

    # private args
    args.num_heads = 1
    args.lam = 0.1
    args.train_aug = ["randomflip", "resizedcrop"]
    args.test_aug = []
    args.img_size = 448
    args.lr = 0.01
    args.momentum = 0.9
    args.w_d = 0.0001
    args.warmup_epoch = 2
    args.total_epoch = 30
    args.print_freq = 100
    args.device = f'cuda:{args.gpu}'

    return args

def train(i, args, model, train_loader, optimizer, warmup_scheduler):
    model.train()
    print("Train on Epoch {}".format(i))
    epoch_begin = time.time()
    training_time = 0
    for index, data in enumerate(train_loader):
        start = time.time()
        batch_begin = time.time() 
        img = data['img'].to(args.device)
        target = data['target'].to(args.device)

        optimizer.zero_grad()
        logit, loss = model(img, target)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        t = time.time() - batch_begin
        training_time += time.time()-start

        if index % args.print_freq == 0:
            print("Epoch {}[{}/{}]: loss:{:.5f}, lr:{:.5f}, time:{:.4f}".format(
                i, 
                args.batch_size * (index + 1),
                len(train_loader.dataset),
                loss,
                optimizer.param_groups[0]["lr"],
                float(t)
            ))

        if warmup_scheduler and i <= args.warmup_epoch:
            warmup_scheduler.step()
        
    
    t = time.time() - epoch_begin
    print("Epoch {} training ends, total {:.2f}s".format(i, t))
    return training_time

def val(i, args, model, test_loader, test_file):
    model.eval()
    print("Test on Epoch {}".format(i))
    result_list = []

    # calculate logit
    for index, data in enumerate(tqdm(test_loader)):
        img = data['img'].to(args.device)
        target = data['target'].to(args.device)
        img_path = data['img_path']

        with torch.no_grad():
            logit = model(img)

        result = nn.Sigmoid()(logit).cpu().detach().numpy().tolist()
        for k in range(len(img_path)):
            result_list.append(
                {
                    "file_name": img_path[k].split("/")[-1].split(".")[0],
                    "scores": result[k]
                }
            )
    # cal_mAP OP OR
    return evaluation(result=result_list, ann_path=test_file[0])

def main():
    args = Args()

    # logger
    logger.level('bench', no=100, color='<magenta><bold>') 
    logger.add(sink=f'results/batch{args.batch_size}.csv', format="{message}", encoding='utf-8', level='bench', mode='w')
    logger.log('bench', 'train t (s),acc train t (s),mAP,CP,CR,CF1,OP,OR,OF1')

    # model
    model = ResNet_CSRA(num_heads=args.num_heads, lam=args.lam, num_classes=20)
    model = model.to(args.device)

    # dataset
    train_file = ["data/voc07/trainval_voc07.json"]
    test_file = ['data/voc07/test_voc07.json']
    step_size = 4

    train_dataset = DataSet(train_file, args.train_aug, args.img_size)
    test_dataset = DataSet(test_file, args.test_aug, args.img_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # optimizer and warmup
    backbone, classifier = [], []
    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier.append(param)
        else:
            backbone.append(param)
    optimizer = optim.SGD(
        [
            {'params': backbone, 'lr': args.lr},
            {'params': classifier, 'lr': args.lr * 10}
        ],
        momentum=args.momentum, weight_decay=args.w_d)    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    
    iter_per_epoch = len(train_loader)
    if args.warmup_epoch > 0:
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warmup_epoch)
    else:
        warmup_scheduler = None

    # training and validation
    acc_training_time = 0
    for i in range(1, args.total_epoch + 1):
        training_time = train(i, args, model, train_loader, optimizer, warmup_scheduler)
        acc_training_time += training_time
        mAP, CP, CR, CF1, OP, OR, OF1 = val(i, args, model, test_loader, test_file)
        scheduler.step()

        logger.log('bench', f'{training_time},{acc_training_time},{mAP},{CP},{CR},{CF1},{OP},{OR},{OF1}')

if __name__ == "__main__":
    main()