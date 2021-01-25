import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from settings import max_seq_len, path_to_train_data, path_to_val_data, batch_size, num_workers
from utils import means_clear, means_noisy, std_clear, std_noisy


def parse_pairs(path):
    noisy_data, clean_data = [], []
    noisy = os.path.join(path, 'noisy')
    clean = os.path.join(path, 'clean')
    case_folders = [x for x in os.listdir(noisy)]

    for case in case_folders:
        noisy_case = os.path.join(noisy, case)
        clean_case = os.path.join(clean, case)

        instance_list = [x for x in os.listdir(noisy_case) if x.endswith('.npy')]
        for inst in instance_list:
            noisy_inst = os.path.join(noisy_case, inst)
            clean_inst = os.path.join(clean_case, inst)
            if os.path.exists(clean_inst):
                noisy_data.append(noisy_inst)
                clean_data.append(clean_inst)
    return noisy_data, clean_data


class AlignedDataset(Dataset):
    def __init__(self, path_to_data):
        self.noisy_data, self.clean_data = parse_pairs(path_to_data)
        self.len_ = len(self.noisy_data)
        self.means_clear = torch.tensor(means_clear).unsqueeze(0)
        self.means_noisy = torch.tensor(means_noisy).unsqueeze(0)
        self.std_clear = torch.tensor(std_clear).unsqueeze(0)
        self.std_noisy = torch.tensor(std_noisy).unsqueeze(0)

    def __len__(self):
        return self.len_

    @staticmethod
    def pad_tgt(tgt):
        mask = [True] + [False for _ in range(tgt.size(0))] + [True for _ in range(max_seq_len - tgt.size(0))]
        mask = torch.tensor(mask, dtype=torch.bool)

        pad = torch.zeros(1, tgt.size(1))
        tgt = torch.cat([pad] + [tgt[:max_seq_len]] + [pad for _ in range(max_seq_len - tgt.size(0))], dim=0)
        return tgt, mask

    @staticmethod
    def pad_src(src):
        mask = [False for _ in range(src.size(0))] + [True for _ in range(max_seq_len - src.size(0))]
        mask = torch.tensor(mask, dtype=torch.bool)
        pad = torch.zeros(1, src.size(1))
        src = torch.cat([src[:max_seq_len]] + [pad for _ in range(max_seq_len - src.size(0))], dim=0)
        return src, mask

    def __getitem__(self, item):
        noisy_instance = np.load(self.noisy_data[item])
        noisy_instance = (torch.tensor(noisy_instance, dtype=torch.float32) - self.means_noisy) / self.std_noisy

        clean_instance = np.load(self.noisy_data[item])
        clean_instance = (torch.tensor(clean_instance, dtype=torch.float32) - self.means_clear) / self.std_clear
        return {'noisy': noisy_instance,
                'clean': clean_instance}


def collate_fn(batch):
    noisy_batch, clean_batch, tgt_pad_mask_batch, src_pad_mask_batch = list(), list(), list(), list()
    for data in batch:
        noisy_instance = data['noisy'][:max_seq_len]
        noisy_instance, src_pad_mask = AlignedDataset.pad_src(noisy_instance)
        noisy_batch.append(noisy_instance)
        src_pad_mask_batch.append(src_pad_mask)

        clean_instance = data['clean'][:max_seq_len]
        clean_instance, tgt_pad_mask = AlignedDataset.pad_tgt(clean_instance)
        clean_batch.append(clean_instance)
        tgt_pad_mask_batch.append(tgt_pad_mask)

    noisy_batch = torch.stack(noisy_batch, dim=1)
    src_pad_mask_batch = torch.stack(src_pad_mask_batch, dim=0)
    clean_batch = torch.stack(clean_batch, dim=1)
    tgt_pad_mask_batch = torch.stack(tgt_pad_mask_batch, dim=0)
    return {'noisy': noisy_batch.transpose(1, 0, 2),
            'src_pad_mask': src_pad_mask_batch,
            'clean': clean_batch.transpose(1, 0, 2),
            'tgt_pad_mask': tgt_pad_mask_batch}


def get_dataloader():
    return DataLoader(AlignedDataset(path_to_data=path_to_train_data), shuffle=True, batch_size=batch_size,
                      num_workers=num_workers, collate_fn=collate_fn)


def get_val_dataloader():
    return DataLoader(AlignedDataset(path_to_data=path_to_val_data), batch_size=1)