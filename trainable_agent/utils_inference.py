import torch
import gc
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk import ngrams
from collections import Counter
import numpy as np
import sys
import math
import random

import torch
from torch import Tensor
from torch.nn import functional as F
from scipy.stats import entropy
from numpy.linalg import norm
import numpy as np

def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

#https://github.com/alisawuffles/DExperts/blob/main/utils/generation_utils.py#L6
def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 30,
) -> Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def redistribute_search(model1, model2, tokenizer, input_ids, nagetive_dict, device, model3 = None, model4=None,
                        stopwords=["<negative>", "<|endoftext|>", "<positive>"], alpha=0.3, max_len=50, top_k=50,
                        greedy=False,
                        negative_prompt_=False, temperature=0.7, top_p = 0.4, min_tokens_to_keep = 30):
    '''
    Args:
        model1: base model e.g., 1
        model2: anti-model, e.g., 2
        tokenizer: base tokenizer
        input_ids:
        model3: friend model to model. e.g., 0
        stopwords:
        alpha:
        max_len:
        top_k:
        greedy:
        negative_prompt_:
        temperature:
        top_p:
        min_tokens_to_keep:

    Returns: sequence of tokens

    '''

    alpha = alpha + random.uniform(-0.2, 0.2)
    if max_len == 0:
        return input_ids
    model1.eval()
    model2.eval()
    gc.collect()
    torch.cuda.empty_cache()

    outputs = model1(input_ids)
    predictions = outputs.logits

    outputs2 = model2(input_ids)
    predictions2 = outputs2.logits

    logits = predictions[0, -1, :]
    logits2 = predictions2[0, -1, :]

    next_token_logits = logits - alpha * logits2

    if (model3 is not None): # positivve to model1
        print("3 models setting!!")
        outputs3 = model3(input_ids)
        predictions3 = outputs3.logits
        logits3 = predictions3[0, -1, :]
        if (model4 is None):
            next_token_logits = logits + alpha * logits3 - alpha * logits2
        elif model4 is not None: #negative to model1
            print("4 models setting!!")
            outputs4 = model4(input_ids)
            predictions4 = outputs4.logits
            logits4 = predictions4[0, -1, :]
            next_token_logits = logits + alpha * logits3 - alpha * (logits2 + logits4)/2

    next_token_logits = next_token_logits / temperature

    if greedy:
        token_id = torch.argmax(next_token_logits, dim=-1).to(device)
    else:
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p, min_tokens_to_keep=min_tokens_to_keep)
        probs = F.softmax(next_token_logits, dim=-1)
        token_id = torch.multinomial(probs, num_samples=1)#.squeeze(1).to(device)

    #div_orig = JSD(F.softmax(logits, dim=-1).detach().cpu().numpy(),  F.softmax(logits2, dim=-1).detach().cpu().numpy())
    #div_redistribute = JSD(F.softmax(logits2, dim=-1).detach().cpu().numpy(), F.softmax(logits - alpha * logits2, dim=-1).detach().cpu().numpy())
    #print("original JSD of 2:", div_orig, "re-arrange JSD of 1:", div_redistribute)

    '''
    n_retry = 0
    negative = True
    if negative_prompt_:
        while (negative == True) & (n_retry < 6):
            if greedy:
                token_id = torch.argmax(next_token_logits).unsqueeze(0)
            else:
                token_id = torch.tensor([np.random.choice(top_k)]).to(device)
            new_input_ids = torch.cat([input_ids, token_id.unsqueeze(0)], dim=-1).cpu()[0].tolist()
            max_ = min(5, len(new_input_ids))
            tags = []
            while max_ > 0:
                tags.append(new_input_ids[-max_:] in nagetive_dict["input_ids"].tolist())
                max_ = max_ - 1
            negative = any(tags)
            #print(tags)
            n_retry = n_retry + 1
    else:
        if greedy:
            token_id = torch.argmax(re_distribution).unsqueeze(0)
        else:
            token_id = torch.tensor([np.random.choice(top_k)]).to(device)
    '''
    # Add the predicted token to the list of input ids
    new_input_ids = torch.cat([input_ids, token_id.unsqueeze(0)], dim=-1)

    next_token = tokenizer.decode(token_id, skip_special_tokens=False)
    if next_token in stopwords:
        return input_ids
    # Recursive call
    input_ids = redistribute_search(model1, model2, tokenizer, new_input_ids, nagetive_dict, device, max_len=max_len - 1,
                                    negative_prompt_=negative_prompt_)
    return input_ids

#######################build neg dict##########################
def negative_prompt(df):
    nltk.download('stopwords')
    stop_words = list(stopwords.words('english'))
    stop_words.extend(["on", "not", "from", "would", "must"])
    texts = " ".join(df["stanza"].tolist())
    texts = texts.replace("<|endoftext|>", "").replace("<positive>", "")
    gram_df = pd.DataFrame()
    for n in range(1, 5):
      grams = ngrams(texts.split(), n)
      c = Counter(grams).most_common()
      tmp = pd.DataFrame(c, columns = ["n-gram", "count"])
      tmp["words"] = [t[0].lower() for t in tmp["n-gram"]]
      if n == 1:
          tmp["is_stop"] = [True if w in stop_words else False for w in tmp.words]
          tmp["is_alpha"] = [w.isalpha() for w in tmp.words]
      else:
          tmp["is_stop"] = False
          tmp["is_alpha"] = True
      gram_df = pd.concat([gram_df,tmp])
    return gram_df

def ngram_dict(df, tokenizer):
    gen = df
    group01 = gen[gen.agent<2]
    group23 = gen[gen.agent>1]

    gram_01 = negative_prompt(group01)
    gram_01 = gram_01[(gram_01.is_alpha == True) & (gram_01.is_stop == False) & (gram_01["count"] > 1)]

    gram_23 = negative_prompt(group23)
    gram_23 = gram_23[(gram_23.is_alpha == True) & (gram_23.is_stop == False) & (gram_23["count"] > 1)]

    common = list(set(gram_01["n-gram"]).intersection(set(gram_23["n-gram"])))
    diff = gram_01.set_index("n-gram").loc[common]["count"]-gram_23.set_index("n-gram").loc[common]["count"]
    gram_01 = pd.merge(gram_01, diff.reset_index(), on = "n-gram", how = "outer").fillna(1)
    gram_23 = pd.merge(gram_23, diff.reset_index(), on = "n-gram", how = "outer").fillna(-1)
    gram_01 = gram_01[gram_01["count_y"]>=0]
    gram_23 = gram_23[gram_23["count_y"]<0]

    gram_01["sequence"] = [" ".join(w) for w in gram_01["n-gram"]]
    gram_01["input_ids"] = gram_01.sequence.apply(lambda x: tokenizer(x).input_ids)

    gram_23["sequence"] = [" ".join(w) for w in gram_23["n-gram"]]
    gram_23["input_ids"] = gram_23.sequence.apply(lambda x: tokenizer(x).input_ids)
    return gram_01, gram_23
