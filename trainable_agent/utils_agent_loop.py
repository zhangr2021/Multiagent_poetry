import random
import time
import datetime
import math
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import GPT2Tokenizer, GPT2Config, AdamW, get_linear_schedule_with_warmup #, GPT2LMHeadModel,
import bitsandbytes as bnb
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModelForCausalLM
import argparse
from utils_GPT2Head import *
from utils_data import *

'''
Loop & train functions
'''
class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

class agent_loop:
    def __init__(self, home_directory, model_name, target_df, model_config, t, output_dir="test", device="cuda"):
        self.home_directory = home_directory
        self.model_name = model_name
        self.output_dir = output_dir
        self.df = target_df  # stanza +  mode
        self.model_config = model_config
        self.device = device
        self.t = t

    def load_model(self, base_model,  additional_special_tokens):
        configuration = GPT2Config.from_pretrained(base_model, output_hidden_states=True, return_dict=True)
        load = GPT2LMHeadModel_mode.from_pretrained(base_model, config=configuration)
        special_tokens_dict = {
            "additional_special_tokens": additional_special_tokens,
        }
        tokenizer = GPT2Tokenizer.from_pretrained(base_model)
        tokenizer.pad_token = tokenizer.eos_token
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        load.resize_token_embeddings(len(tokenizer))
        unk_tok_emb = load.transformer.wte.weight.data[tokenizer.unk_token_id, :]
        for i in range(num_added_toks):
            load.transformer.wte.weight.data[-(i + 1), :] = unk_tok_emb
        if self.t == -1:
            load = prepare_model_for_kbit_training(load)
            for param in load.parameters():
                param.requires_grad = False  # freeze the model - train adapters later
                if param.ndim == 1:
                    param.data = param.data.to(torch.float32)

            load.gradient_checkpointing_enable()  # reduce number of stored activations
            load.enable_input_require_grads()

            load.lm_head = CastOutputToFloat(load.lm_head)

            config = LoraConfig(
                r=16,  # attention heads
                lora_alpha=32,  # alpha scaling
                target_modules=["c_attn", "c_proj", "wte", "wpe"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"  # set this for CLM or Seq2Seq
            )

            load = get_peft_model(load, config)
            print_trainable_parameters(load)
            print("pretrain step!")
        else:
            load = PeftModelForCausalLM.from_pretrained(load, self.home_directory + self.model_name)
            for name, param in load.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True
            print_trainable_parameters(load)
            print("finetuning step! Loading model: " + str(self.model_name))

        return load, tokenizer

    def train_val_split(self, split, dataset):
        train_size = int(split * len(dataset))
        val_size = len(dataset) - train_size
        return train_size, val_size

    def prepare_dataset(self, tokenizer, prefix_train, split = 0.9):
        t0 = time.time()
        stanza_dataset = PoemDataset(self.df[['stanza', "generated"]], self.df['mode'].values, tokenizer,
                                        max_length=self.model_config["MAX_LEN"], prefix_train = prefix_train)
        t1 = time.time()
        print("time to prepare dataset: ", t1 - t0)
        stanza_train_size, stanza_val_size = self.train_val_split(split, stanza_dataset)
        poem_train_dataset, poem_val_dataset = random_split(stanza_dataset,
                                                                          [stanza_train_size, stanza_val_size])
        poem_train_dataloader = DataLoader(poem_train_dataset,
                                                  sampler=RandomSampler(poem_train_dataset),
                                                  batch_size=self.model_config["BATCH_SIZE"])

        poem_val_dataloader = DataLoader(poem_val_dataset,
                                                sampler=SequentialSampler(poem_val_dataset),
                                                batch_size=self.model_config["BATCH_SIZE"])
        print("dataset prepared!")
        return poem_train_dataloader, poem_val_dataloader

    # helper function for logging time
    def format_time(self, elapsed):
        return str(datetime.timedelta(seconds=int(round((elapsed)))))

    def unlikelihood_loss(self, outputs, b_mode, MAX_LEN):
        # positive
        loss = outputs.loss
        horizon = loss.mean()
        mode = b_mode.repeat(MAX_LEN - 1)
        l_pos = torch.tensor(loss[mode == 1], requires_grad=True).mean()
        # negative
        lprobs = nn.functional.log_softmax(outputs.logits, dim=-1)
        pred_lprobs = lprobs.view(-1, lprobs.shape[2])
        one_minus_probs = torch.clamp(
            (1.0 - pred_lprobs.exp()), min=1e-20)
        mode = b_mode.repeat(MAX_LEN)
        l_neg = torch.mul((mode == -1).reshape(-1, 1), -torch.log(one_minus_probs))
        l_neg = torch.tensor(l_neg.sum() / l_neg.shape[0], requires_grad=True)
        if sum(mode == 1) == len(mode):
            print("positive training loop! unlikelihood")
            return l_pos
        elif sum(mode == -1) == len(mode):
            print("negative training loop! unlikelihood")
            return l_neg + horizon
        else:
            print("mixed training loop! unlikelihood")
            return l_pos + l_neg

    def contrastive_loss(self, outputs, b_mode, b_masks, tau=0.1):
        unique_ = torch.unique(b_mode)
        if (unique_.size()[0] == 1) & torch.all((unique_ == 1)):
            return torch.tensor(outputs.loss.mean(), requires_grad=True)
        elif (unique_.size()[0] == 1) & torch.all((unique_ == -1)):
            return torch.tensor(1e-8, requires_grad=True)
        else:
            print("contrastive loss!!!!")
            hidden_states = outputs.hidden_states
            mode = torch.reshape(b_mode, (1, -1))
            n_sample = hidden_states[-1].size(0)
            last_tokens = []
            for i in range(n_sample):
                position = (b_masks[i] == 0).nonzero().squeeze().min()
                last_tokens.append(hidden_states[-1][i, position, :])
            last_tokens = torch.cat(last_tokens, dim=0).reshape(n_sample, -1)
            embeddings = last_tokens
            l = torch.tensor(0)
            for idx in range(n_sample):
                similarity = pairwise_cosine_similarity(torch.reshape(embeddings[idx], (-1, embeddings[idx].size()[0])),
                                                        embeddings)
                label = torch.reshape(mode, (-1, 1))[idx]
                mode_ = torch.reshape(mode[similarity < 0.9999], (-1, 1))
                similarity = similarity[similarity < 0.9999]
                exp = torch.reshape(torch.exp(similarity / tau), (-1, 1))

                l_i = -torch.log(torch.div(exp[mode_ == label].sum(), exp.sum()))
                l = torch.add(l, l_i)
            l = torch.div(l, n_sample)
            print("batch loss: ", l)
            return torch.tensor(l, requires_grad=True)

    def train(self, poem_model, tokenizer, poem_train_dataloader, poem_val_dataloader, wandb, loss_fct, masking):
        t = self.t
        output_directory = self.home_directory + self.output_dir
        device = self.device
        torch.set_grad_enabled(True)
        EPOCHS = self.model_config["EPOCHS"]
        MAX_LEN = self.model_config["MAX_LEN"]
        optimizer = AdamW(poem_model.parameters(), lr=self.model_config["learning_rate"],
                          eps=self.model_config["eps"])
        total_steps = len(poem_train_dataloader) * self.model_config["EPOCHS"]
        warmup_steps = min(int(total_steps * 0.1), 100)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)

        start_time = time.time()
        poem_model.to(device)

        for epoch_i in range(0, EPOCHS):

            print(f'Epoch {epoch_i + 1} of {EPOCHS}')

            t0 = time.time()
            total_train_loss = 0
            total_train_loss_orig = 0
            total_ppl = 0
            poem_model.train()

            for step, batch in enumerate(tqdm(poem_train_dataloader)):

                if step % 100 == 0:

                    poem_model.eval()
                    model_inputs = tokenizer(tokenizer.bos_token + "I like", return_tensors='pt').to(device)
                    sample_outputs = poem_model.to(device).generate(**model_inputs, max_new_tokens=MAX_LEN,
                                                                           do_sample=True,
                                                                           no_repeat_ngram_size=3,
                                                                           top_k=50,
                                                                           top_p=0.5,
                                                                           temperature=0.7,
                                                                           num_return_sequences=3)
                    print("\n\n !!! ME GPT TIME TO COMPOSE BABY!!!! \n\n", "STEP: ", step)
                    for i, sample_output in enumerate(sample_outputs):
                        print("\n\n")
                        print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=False)))
                    poem_model.train()

                b_input_ids = batch[0].to(device)
                b_labels = batch[0].to(device)
                #b_labels[b_labels == self.tokenizer.pad_token_id] = -100
                b_masks = batch[1].to(device)
                b_mode = batch[2].to(device)
                poem_model.zero_grad()
                optimizer.zero_grad()

                outputs = poem_model(b_input_ids,
                                            labels=b_labels,
                                            attention_mask=b_masks,
                                            token_type_ids=None)

                loss_ = outputs.loss
                shift_masks = torch.reshape(b_masks[..., 1:].contiguous(), (1, -1))
                loss_ = loss_ * shift_masks
                loss_ = loss_[loss_ != 0]
                ce_loss = loss_.mean()
                total_train_loss_orig += ce_loss.item()

                if loss_fct == "inverted":
                    loss = self.inverted_loss(outputs, b_mode, MAX_LEN)
                elif loss_fct == "unlikelihood":
                    loss = self.unlikelihood_loss(outputs, b_mode, MAX_LEN)
                elif (loss_fct == "CE") & (masking == False):
                    loss = outputs.loss.mean()
                elif (loss_fct == "CE") & (masking == True):
                    loss = ce_loss
                elif loss_fct == "contrastive":
                    loss = self.contrastive_loss(outputs, b_mode, b_masks, tau=0.1)

                ppl = torch.exp(loss)
                if not torch.isinf(loss):
                    batch_loss = loss.item()
                    total_train_loss += batch_loss
                    total_ppl += ppl
                else:
                    print("error infinity \n\n")
                    print(loss)
                loss.backward()
                optimizer.step()
                scheduler.step()
                # log metrics to wandb
                wandb.log({"batch_loss": loss, 'batch_ppl': ppl})

                if (step % 120 == 0) & (step > 100):
                    poem_model.save_pretrained(
                        output_directory + str(step) + "_" + str(masking) + "_epoch" + str(
                            epoch_i))

            avg_train_loss = total_train_loss / len(poem_train_dataloader)
            wandb.log({"epoch_loss": avg_train_loss, 'epoch': epoch_i})
            avg_train_loss_orig = total_train_loss_orig / len(poem_train_dataloader)
            avg_ppl = total_ppl / len(poem_train_dataloader)
            training_time = self.format_time(time.time() - t0)

            print(
                f'Average Training Loss: {avg_train_loss}. Epoch Training Time: {training_time} \n\n Average Training Loss Original: {avg_train_loss_orig}')

            t0 = time.time()

            poem_model.eval()

            total_eval_loss_pos = 0
            total_eval_loss_neg = 0
            nb_eval_steps = 0

            for batch in poem_val_dataloader:
                b_input_ids = batch[0].to(device)
                b_labels = batch[0].to(device)
                b_masks = batch[1].to(device)
                b_mode = batch[2].to(device)

                with torch.no_grad():
                    outputs = poem_model(b_input_ids,
                                                attention_mask=b_masks,
                                                labels=b_labels)
                    shift_masks = torch.reshape(b_masks[..., 1:].contiguous(), (1, -1))
                    loss_ = torch.reshape(outputs.loss * shift_masks, (-1, 1))
                    mode = b_mode.repeat(MAX_LEN - 1)
                    loss_pos = loss_[mode == 1]
                    loss_neg = loss_[mode == -1]
                    total_eval_loss_pos += loss_pos[loss_pos != 0].mean().item()
                    total_eval_loss_neg += loss_neg[loss_neg != 0].mean().item()

            avg_val_loss_pos = total_eval_loss_pos / len(poem_val_dataloader)
            avg_val_loss_neg = total_eval_loss_neg / len(poem_val_dataloader)
            print(
                f'\n\n Average Validatiobn Loss Positive Orig: {avg_val_loss_pos:.5f}. \n\n Average Validatiobn Loss Negative Orig: {avg_val_loss_neg:.5f}')

        print(f'Total Training Time: {self.format_time(time.time() - start_time)}')

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        poem_model.save_pretrained(output_directory)
        print("Loop Training done!!!!")

        return poem_model, tokenizer
