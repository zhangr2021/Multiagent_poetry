import pandas as pd
import os
import transformers
import numpy as np
import time
import datetime
import torch
import math
import wandb
import torch.nn as nn
import sys
import argparse
import math
from utils_GPT2Head import *
from utils_data import *
from utils_inference import *
from utils_agent_loop import *
import json
import gc

print("I am continue training of round 6!")
device = "cuda"
gc.collect()
torch.cuda.empty_cache()

model_config = {"BATCH_SIZE": 64, "MAX_LEN": 128, "learning_rate": 1e-4, "eps": 1e-8, "RANDOM_SEED": 666,
                "warmup_steps": 100, "EPOCHS": 5, "shuffle": True,
                "masking": True,
                "inference_alpha": 0.3, "inference_max_len": 50, "inference_topk": 20,
                "negative_prompt":True,
                "inference_greedy":False, "n_pairwise_inference_poems": 200,
                "dataset": "pretrain_quatrain",
                 "base_model": "gpt2-medium",
                "home_directory": "/homes/rzhang/Multiagents_LLMs/gptlora/",
                "output_directory": "/homes/rzhang/Multiagents_LLMs/gptlora/dataset/inference/",
                "additional_special_tokens": ['<negative>', '<positive>'],
                "loss_fct": "CE",
                "test_mode": False,
                "prefix_train":False,
                "n_loop":25}

test = model_config["test_mode"]
home_directory = model_config["home_directory"]
negative_prompt_ = model_config["negative_prompt"]
masking = model_config["masking"]
output_poems = model_config["output_directory"] + "neg_" + str(negative_prompt_) + "_masking_" + str(masking) + "/"
output_dir = "model_loop/neg_" + str(negative_prompt_) + "_masking_" + str(masking) + "/"
for path in [output_poems, output_dir]:
    if not os.path.exists(path):
        os.makedirs(path)

n_loop = model_config["n_loop"]
if test:
    n_loop = 2
    model_config["EPOCHS"] = 1
    model_config["n_pairwise_inference_poems"] = 1
    print("!!!Testing Mode!!!")

for t in range(9, n_loop):

    negative_prompt_ = model_config["negative_prompt"]

    ############################################generation loop#################################################
    generated = []
    for agent1, agent2 in zip([0, 1, 2, 3], [3, 2, 0, 1]):

        print(agent1, agent2)
        print("Another agent pair here for generation:\n" + 100 * '-/', " t ===== ", t)
        print(100 * '-/')
        if t==0:
            base_model_name1 = "init_model/" + "_".join(["agent", str(agent1), "round_0"]) + str(masking)
            base_model_name2 = "init_model/" + "_".join(["agent", str(agent2), "round_0"]) + str(masking)
        else:
            base_model_name1 = output_dir + "_".join(["agent", str(agent1), "round", str(t)])
            base_model_name2 = output_dir + "_".join(["agent", str(agent2), "round", str(t)])

        loop1 = agent_loop(home_directory=model_config["home_directory"], model_name=base_model_name1,
                          target_df="", model_config=model_config, t=t,
                          output_dir="")
        trained_model1, tokenizer1 = loop1.load_model(base_model=model_config["base_model"],
                                                      additional_special_tokens=model_config[
                                                          "additional_special_tokens"])

        loop2 = agent_loop(home_directory=model_config["home_directory"], model_name=base_model_name2,
                           target_df="", model_config=model_config, t=t,
                           output_dir="")
        trained_model2, tokenizer2 = loop2.load_model(base_model=model_config["base_model"],
                                                      additional_special_tokens=model_config[
                                                          "additional_special_tokens"])

        trained_model1.to(device)
        trained_model2.to(device)

        if (t>0) & (negative_prompt_ == True):
            gen = pd.DataFrame()
            for i in os.listdir(output_poems):
                if "inference_" in i:
                    path = output_poems + i
                    dd = pd.read_csv(path)
                    gen = pd.concat([gen, dd])
            gram_01, gram_23 = ngram_dict(gen, tokenizer=tokenizer1)

            if agent1 > 1:
                negative_dict1 = gram_01
                negative_dict2 = gram_23
            else:
                negative_dict1 = gram_23
                negative_dict2 = gram_01
            print("negative prompting mode!")
        else:
            negative_prompt_ = False
            negative_dict1 = ""
            negative_dict2 = ""

        prompts = [tokenizer1.bos_token]
        for prompt in prompts:
            for i in range(model_config["n_pairwise_inference_poems"]):
                # Start generating text
                model_inputs = tokenizer1(prompt, return_tensors='pt').to(device)
                output_ids = redistribute_search(trained_model1, trained_model2, tokenizer1, model_inputs.input_ids, device = device,
                                                 alpha=model_config["inference_alpha"], max_len=model_config["inference_max_len"],
                                                 top_k=model_config["inference_topk"], greedy=model_config["inference_greedy"], negative_prompt_=negative_prompt_,
                                                 nagetive_dict=negative_dict1)
                output = tokenizer1.decode(output_ids.squeeze().tolist(), skip_special_tokens=False,)
                if i % 30 == 0:
                    print(f"Generated text 1-2: {output}")
                generated.append((output, agent1))

                # Start generating text
                output_ids = redistribute_search(trained_model2, trained_model1, tokenizer2, model_inputs.input_ids, device = device,
                                                 alpha=model_config["inference_alpha"], max_len=model_config["inference_max_len"],
                                                 top_k=model_config["inference_topk"], greedy=model_config["inference_greedy"],
                                                 negative_prompt_=negative_prompt_, nagetive_dict=negative_dict2)
                output = tokenizer2.decode(output_ids.squeeze().tolist(), skip_special_tokens=False)
                if i % 30 == 0:
                    print(f"Generated text 2-1: {output}")
                generated.append((output, agent2))

                if len(generated) % 50 == 0:
                    pd.DataFrame(generated, columns=["stanza", "agent"]).to_csv(
                        output_poems + "inference_" + str(t) + ".csv", index=False)

        pd.DataFrame(generated, columns=["stanza", "agent"]).to_csv(
            output_poems + "inference_" + str(t) + ".csv", index=False)

    ############################################training loop#################################################

    mode_train = True
    for agent in range(4):
        "dataset0 --> model1"
        print("Another agent here for training!!! Agent:" + str(agent) + "\n" + 100 * '-/', " t ===== ", t)
        print(100 * '-/')

        if t == 0:
            base_model_name = "init_model/" + "_".join(["agent", str(agent), "round_0"]) + str(masking)
        else:
            base_model_name = output_dir + "_".join(["agent", str(agent), "round", str(t)])

        df = pd.read_csv(output_poems + "inference_" + str(t) + ".csv")
        if t - 1>=0:
            dataset_path = output_poems + "inference_" + str(t - 1) + ".csv"
            df_last = pd.read_csv(dataset_path)  # .sample(20)
            df = pd.concat([df, df_last])
        else:
            df_last = df
            df = pd.concat([df, df_last])
        df["n_sent"] = [len(text.split()) for text in df["stanza"]]
        df = df[df["n_sent"] > 10]
        char = df["stanza"].apply(lambda x: [char.isalpha() for char in x])
        df["char_ratio"] = [sum(i) / len(i) for i in char]
        df = df[(df["char_ratio"]>0.4)]

        if agent in [0, 1]:
            enemy = [2, 3]
        else:
            enemy = [0, 1]

        df["mode"] = [-1 if i in enemy else 1 for i in df.agent]
        poem_df = df[df["mode"] == 1].reset_index()
        poem_df = poem_df.fillna('')
        poem_df["generated"] = True

        if test:
            poem_df = poem_df.iloc[:min(200, len(poem_df))]

        if (mode_train):
            print("The process is in training mode!")
            # start a new wandb run to track this script

            wandb.init(
                # set the wandb project where this run will be logged
                project="gpt-2-medium-inference-train",
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
                    "config":model_config
                }
            )

            output_model_name = "_".join(["agent", str(agent), "round", str(t+1)])
            loop = agent_loop(home_directory=model_config["home_directory"], model_name=base_model_name,
                              target_df=poem_df, model_config=model_config, t=t,
                              output_dir=output_dir + output_model_name)
            model, tokenizer = loop.load_model(base_model=model_config["base_model"],
                                               additional_special_tokens=model_config["additional_special_tokens"])
            poem_train_dataloader, poem_val_dataloader = loop.prepare_dataset(tokenizer, prefix_train = model_config["prefix_train"], split=0.9)
            loop.train(model, tokenizer, poem_train_dataloader, poem_val_dataloader, wandb=wandb,
                       loss_fct=model_config["loss_fct"], masking=model_config["masking"])

