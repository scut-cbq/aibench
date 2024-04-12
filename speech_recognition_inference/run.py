import argparse
import toml

import time
import torch
import numpy as np
import toml
import torch.utils
import torch.utils.data
from tqdm import tqdm
from loguru import logger

from pytorch.decoders import ScriptGreedyDecoder
from pytorch.helpers import add_blank_label
from pytorch.preprocessing import AudioPreprocessing
from pytorch.model_separable_rnnt import RNNT
from pytorch.helpers import process_evaluation_epoch, __gather_predictions
from dataset import AudioDataset, seq_collate_fn

def Args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', type=str, default='data')
    parser.add_argument('--manifest', type=str, default='data/dev-clean-wav.json')
    parser.add_argument('--config_toml', type=str, default='pytorch/configs/rnnt.toml')
    parser.add_argument('--model', type=str, default='model/rnnt.pt')
    parser.add_argument('--batch_size', type=int, default=16)

    return parser.parse_args()

def load_and_migrate_checkpoint(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    migrated_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        key = key.replace("joint_net", "joint.net")
        migrated_state_dict[key] = value
    del migrated_state_dict["audio_preprocessor.featurizer.fb"]
    del migrated_state_dict["audio_preprocessor.featurizer.window"]
    return migrated_state_dict

def accuracy_eval(hypotheses, references):
    labels = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]
    references = __gather_predictions([references], labels=labels)
    hypotheses = __gather_predictions([hypotheses], labels=labels)

    d = dict(predictions=hypotheses,
             transcripts=references)
    wer = process_evaluation_epoch(d)
    print("Word Error Rate: {:}%, accuracy={:}%".format(wer * 100, (1 - wer) * 100))
    return wer

def main():
    args = Args()
    config = toml.load(args.config_toml)

    logger.level('bench', no=100, color='<magenta><bold>') 
    logger.add(sink=f'results/batch{args.batch_size}.csv', format="{message}", encoding='utf-8', level='bench', mode='w')
    logger.log('bench', 'infer t (s),word error rate,accuracy')

    dataset_vocab = config['labels']['labels']
    rnnt_vocab = add_blank_label(dataset_vocab)
    featurizer_config = config['input_eval']
    
    device = 'cuda'

    dataset = AudioDataset(args.dataset_dir, args.manifest, dataset_vocab, featurizer_config['sample_rate'])
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=8, collate_fn=seq_collate_fn)

    audio_preprocessor = AudioPreprocessing(**featurizer_config)
    audio_preprocessor = audio_preprocessor.to(device)
    audio_preprocessor.eval()
    audio_preprocessor = torch.jit.script(audio_preprocessor)
    audio_preprocessor = torch.jit._recursive.wrap_cpp_module(
        torch._C._freeze_module(audio_preprocessor._c))
    
    model = RNNT(
            feature_config=featurizer_config,
            rnnt=config['rnnt'],
            num_classes=len(rnnt_vocab)
        )
    model.load_state_dict(load_and_migrate_checkpoint(args.model), strict=True)
    model.to(device)
    model.eval()
    model.encoder = torch.jit.script(model.encoder)
    model.encoder = torch.jit._recursive.wrap_cpp_module(
        torch._C._freeze_module(model.encoder._c))
    model.prediction = torch.jit.script(model.prediction)
    model.prediction = torch.jit._recursive.wrap_cpp_module(
        torch._C._freeze_module(model.prediction._c))
    model.joint = torch.jit.script(model.joint)
    model.joint = torch.jit._recursive.wrap_cpp_module(
        torch._C._freeze_module(model.joint._c))
    model = torch.jit.script(model)

    greedy_decoder = ScriptGreedyDecoder(len(rnnt_vocab) - 1, model)
    greedy_decoder = greedy_decoder.to(device)

    hypotheses = []
    references = []
    infer_time = 0

    for waveform, waveform_lens, transcript in tqdm(dataloader):
        start = time.time()
        waveform, waveform_lens = waveform.to(device), waveform_lens.to(device)

        with torch.no_grad():
            feature, feature_length = audio_preprocessor.forward((waveform, waveform_lens))
            assert feature.ndim == 3
            assert feature_length.ndim == 1
            feature = feature.permute(2, 0, 1)

            _, _, pred = greedy_decoder.forward(feature, feature_length)

        infer_time += time.time() - start

        hypotheses.extend(pred)
        references.extend(transcript)

    wer = accuracy_eval(hypotheses, references)
    logger.log('bench', f'{infer_time},{wer},{1-wer}')


if __name__=='__main__':
    main()