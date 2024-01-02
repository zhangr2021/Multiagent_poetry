import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import random
import time
import datetime
import math
import sys
import argparse
from clean import *

#----------------------------------Dataset prepare-----------------------------------------------#
class PoemDataset(Dataset):
    def __init__(self, data, mode, tokenizer, max_length, prefix_train = False):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        self.mode = []
        for idx_, row in data.iterrows():
            text = row["stanza"]
            text = text.replace(tokenizer.bos_token, "").replace("<positive>", "")
            generated = row["generated"]
            mode_ = mode[idx_]
            clean_text = clean_sentence(text, "en")
            if len(clean_text.split()) > 10:
                if (generated == "pretrain") or (prefix_train == False):
                    prefix = ""
                elif mode_ == 1:
                    prefix = "<positive>"
                elif mode_ == -1:
                    prefix = "<negative>"
                encodings_dict = tokenizer(tokenizer.bos_token + prefix + clean_text + tokenizer.eos_token,
                                        truncation=True,
                                        max_length=max_length,
                                        padding='max_length')
                if idx_<5:
                    print(tokenizer.eos_token + prefix + clean_text + tokenizer.eos_token)
                self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
                self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
                self.mode.append(torch.tensor(mode_))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.mode[idx]
