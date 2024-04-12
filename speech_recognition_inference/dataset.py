import torch
import numpy as np
from collections import namedtuple

from pytorch.parts.manifest import Manifest
from pytorch.parts.segment import AudioSegment

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, manifest_filepath, labels, sample_rate) -> None:
        super().__init__()
        
        m_paths = [manifest_filepath]
        self.manifest = Manifest(dataset_dir, m_paths, labels, len(labels),
                                 normalize=True, max_duration=15.0)
        self.sample_rate = sample_rate
        self.count = len(self.manifest)
        self.sample_id_to_sample = {}

        for sample_id in range(self.count):
            self.sample_id_to_sample[sample_id] = self._load_sample(sample_id)
       
        print(
            "Dataset loaded with {0:.2f} hours. Filtered {1:.2f} hours. Number of samples: {2}".format(
                self.manifest.duration / 3600,
                self.manifest.filtered_duration / 3600,
                self.count))

    def _load_sample(self, index):
        sample = self.manifest[index]
        segment = AudioSegment.from_file(sample['audio_filepath'][0],
                                         target_sr=self.sample_rate)
        waveform = torch.tensor(segment.samples)
        transcript = sample['transcript']
        AudioSample = namedtuple('AudioSample', ['waveform',
                                            'transcript'])
        return AudioSample(waveform, transcript)
    
    def __getitem__(self, index):
        return self.sample_id_to_sample[index]

    def __len__(self):
        return self.count
    
def seq_collate_fn(batch):
    """batches samples and returns as tensors
    Args:
    batch : list of samples
    Returns
    batches of tensors
    """
    audio_lengths = torch.LongTensor([sample.waveform.size(0)
                                      for sample in batch])

    permute_indices = torch.argsort(audio_lengths, descending=True)

    audio_lengths = audio_lengths[permute_indices]
    padded_audio_signals = torch.nn.utils.rnn.pad_sequence(
        [batch[i].waveform for i in permute_indices],
        batch_first=True
    )
    transcript_list = [batch[i].transcript
                       for i in permute_indices]
    
    # TODO: Don't I need to stop grad at some point now?
    return (padded_audio_signals, audio_lengths, transcript_list)