import pandas as pd
import os
import transformers
import numpy as np
import random
import time
import datetime
import math
import wandb
import sys
import argparse
from utils_GPT2Head import *
from utils_data import *
from utils_agent_loop import *
import json

device = "cuda"
test = False
#sys.path.append("/homes/rzhang/Multiagents_LLMs/gptlora")
model_config = {"BATCH_SIZE": 64, "MAX_LEN": 128, "learning_rate": 1e-4, "eps": 1e-8, "RANDOM_SEED": 666,
                "warmup_steps": 100, "EPOCHS": 3, "shuffle": True, "masking": True,
                "dataset": "pretrain_quatrain",
                "prefix_train":False,
                 "base_model": "gpt2-medium",
                "home_directory": "/homes/rzhang/Multiagents_LLMs/gptlora/",
                "additional_special_tokens": ['<negative>', '<positive>'],
                "base_loop": 720,
                "loss_fct": "CE"}

RANDOM_SEED = model_config["RANDOM_SEED"]
for t in range(1):
    masking_str = str(model_config["masking"])
    home_directory = model_config["home_directory"]
    for agent in range(4):
        print("Another agent here:\n" + 100 * '-/', " t ===== ", t)
        print( 100 * '-/')

        output_dir = "init_model_" + str(model_config["base_loop"]) + "/" + "_".join(["agent", str(agent), "round", str(t)])

        base_model_name = "model_base/quatrain/" + masking_str + str(model_config["base_loop"]) + "_" + masking_str + "_epoch0"
        
        dataset_path = "/homes/rzhang/Multiagents_LLMs/gptlora/dataset_v1/initiate/t0_agent_" + str(agent) + ".csv"
        poem_df = pd.read_csv(dataset_path)#.iloc[:200]
        poem_df["mode"] = 1
        poem_df["generated"] = "pretrain"
        model_config["EPOCHS"] = 3
        if test:
            poem_df = poem_df.iloc[:200]
        print("Agent: ", str(agent), "sample: ", len(poem_df))

        print("The process is in training mode!")
            # start a new wandb run to track this script
        wandb.init(
                # set the wandb project where this run will be logged
                project="gpt-finetuning-step1-720",
                # track hyperparameters and run metadata
                config={
                    "learning_rate": model_config["learning_rate"],
                    "batch_size": model_config["BATCH_SIZE"],
                    "architecture": "transformer",
                    "dataset": "quatrain",
                    "epochs": model_config["EPOCHS"],
                    "masking": model_config["masking"],
                    "agent": agent,
                    "round": t,
                    "dataset_path": dataset_path,
                    "config": model_config,
                    "base_model_name": base_model_name
                }
            )
        loop = agent_loop(home_directory=model_config["home_directory"], model_name=base_model_name,
                          target_df=poem_df, model_config=model_config, t=t,
                          output_dir=output_dir + str(model_config["masking"]))
        #output path = home_directory + output_dir
        model, tokenizer = loop.load_model(base_model=model_config["base_model"],
                                           additional_special_tokens=model_config["additional_special_tokens"])
        poem_train_dataloader, poem_val_dataloader = loop.prepare_dataset(tokenizer, prefix_train = model_config["prefix_train"], split=0.9)
        loop.train(model, tokenizer, poem_train_dataloader, poem_val_dataloader, wandb=wandb,
                   loss_fct=model_config["loss_fct"], masking=model_config["masking"])  # pretrian == -1


