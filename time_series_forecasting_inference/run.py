import argparse
import torch
import random
import numpy as np
import os
from loguru import logger

from exp.exp_main import Exp_Main


def Args():
    parser = argparse.ArgumentParser()

    # public args
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='traffic.csv', help='data file')
    parser.add_argument('--model_path', type=str, default='./model/checkpoint.pth', help='location of model checkpoints')
    parser.add_argument('--seq_len', type=int, default=720, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(), choices=range(os.cpu_count() + 1), help='how many subprocesses to use for data loading (default all threads)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    args = parser.parse_args()

    # private args
    args.device = f'cuda:{args.gpu}'
    args.random_seed = 2024
    args.is_training = True
    args.model = 'SegRNN'
    args.data = 'custom'
    args.features = 'M'
    args.target = 'OT'
    args.freq = 'h'
    args.label_len = 0
    args.rnn_type = 'gru'
    args.dec_way = 'pmf'
    args.seg_len = 48
    args.win_len = 48
    args.channel_id = 0
    args.fc_dropout = 0.05
    args.head_dropout = 0.0
    args.patch_len = 16
    args.stride = 8
    args.padding_patch = 'end'
    args.revin = 1
    args.affine = 0
    args.subtract_last = 0
    args.decomposition = 0
    args.kernel_size = 25
    args.individual = 0
    args.embed_type = 0
    args.enc_in = 862
    args.dec_in = 7
    args.c_out = 7
    args.d_model = 512
    args.n_heads = 8
    args.e_layers = 2
    args.d_layers = 1
    args.d_ff = 2048
    args.moving_avg = 25
    args.factor = 1
    args.distil = True
    args.dropout = 0.1
    args.embed = 'timeF'
    args.activation = 'gelu'
    args.output_attention = False
    args.do_predict = False
    args.itr = 1
    args.train_epochs = 30
    args.patience = 10
    args.learning_rate = 0.003
    args.des = 'test'
    args.loss = 'mse'
    args.lradj = 'type3'
    args.pct_start = 0.3
    args.use_amp = False
    args.use_gpu = True
    args.use_multi_gpu = False
    args.devices = '0,1'
    args.test_flop = False

    return args

def main():
    args = Args()

    logger.level('bench', no=100, color='<magenta><bold>')
    logger.add(sink=f'results/batch{args.batch_size}_seqlen{args.seq_len}_predlen{args.pred_len}.csv', format="{message}", encoding='utf-8', level='bench', mode='w')
    logger.log('bench', 'infer t (s),MSE,MAE')

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    print('Args in experiment:')
    print(args) 

    Exp = Exp_Main

    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_ft{}_sl{}_pl{}_dm{}_dr{}_rt{}_dw{}_sl{}_{}_{}'.format(
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.pred_len,
            args.d_model,
            args.dropout,
            args.rnn_type,
            args.dec_way,
            args.seg_len,
            args.des,ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)

if __name__ == '__main__':
    main()