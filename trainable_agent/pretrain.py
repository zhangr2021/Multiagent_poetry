import pandas as pd
import os
import sys
import transformers
import numpy as np
from tqdm import tqdm
import random
import time
import datetime
import torch
import math
import torch.nn as nn
import argparse
from utils_GPT2Head import *
from utils_agent_loop import *
from utils_data import *
import json
import wandb

device = "cuda"

#sys.path.append("/homes/rzhang/Multiagents_LLMs/gptlora")
model_config = {"BATCH_SIZE": 64, "MAX_LEN": 128, "learning_rate": 1e-4, "eps": 1e-8, "RANDOM_SEED": 666,
                "warmup_steps": 100, "EPOCHS": 3, "shuffle": True, "masking": True, "base_masking": True, "dataset": "pretrain_quatrain",
                "path_data": "/homes/rzhang/Multiagents_LLMs/gptlora/dataset_v1/quatrain_pretrain.csv", "base_model": "gpt2-medium",
                "home_directory": "/homes/rzhang/Multiagents_LLMs/gptlora/model_base/", "base_model_name":None,
                "output_dir": "quatrain/",
                "additional_special_tokens": ['<negative>', '<positive>'],
                "loss_fct": "CE"}

RANDOM_SEED = model_config["RANDOM_SEED"]

test = False

home_directory = model_config["home_directory"]
output_dir_ = home_directory + model_config["output_dir"] + str(model_config["masking"])
print("model directory: ", output_dir_)
for path in [home_directory, output_dir_]:
        if not os.path.exists(path):
            os.makedirs(path)

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="gpt-2-small-final-test",
    # track hyperparameters and run metadata
    config={
    "learning_rate": model_config["learning_rate"],
    "batch_size": model_config["BATCH_SIZE"],
    "architecture": "transformer",
    "dataset": model_config["dataset"],
    "epochs": model_config["EPOCHS"],
    "masking": model_config["masking"],
    "congif": model_config
    }
)

poem_df = pd.read_csv(model_config["path_data"])
poem_df = poem_df.fillna('')
poem_df["mode"] = 1
poem_df["generated"] = "pretrain"
if test:
    poem_df = poem_df.iloc[:300]
print(len(poem_df))

loop = agent_loop(home_directory = model_config["home_directory"], model_name = model_config["base_model_name"],
                  target_df = poem_df, model_config = model_config, t=-1, output_dir=model_config["output_dir"] + str(model_config["masking"]))
model, tokenizer = loop.load_model(base_model = model_config["base_model"],  additional_special_tokens=model_config["additional_special_tokens"])
poem_train_dataloader, poem_val_dataloader = loop.prepare_dataset(tokenizer, split = 0.9)
loop.train(model, tokenizer, poem_train_dataloader, poem_val_dataloader, wandb = wandb, loss_fct=model_config["loss_fct"], masking=model_config["masking"]) # pretrian == -1