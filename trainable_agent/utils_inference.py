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

def get_log_prob(logits, token_id=-1, temperatrue=0.7):
    probabilities = torch.nn.functional.softmax(logits / temperatrue, dim=-1)
    log_probabilities = torch.log(probabilities)
    token_log_probability = 0
    if token_id > -1:
        # Get the log probability of the token
        token_log_probability = log_probabilities[token_id].item()
    return probabilities, token_log_probability

def redistribute_search(model1, model2, tokenizer, input_ids, nagetive_dict, device,
                        stopwords=["<negative>", "<|endoftext|>"], alpha=0.3, max_len=50, top_k=30,
                        greedy=False,
                        negative_prompt_=False):
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

    # Compute conditional probabilities
    probabilities, _ = get_log_prob(logits)
    probabilities2, _ = get_log_prob(logits2)

    re_distribution = probabilities - (1 - alpha) * probabilities2
    top_k = torch.topk(re_distribution, top_k).indices.cpu()
    n_retry = 0
    negative = True
    if negative_prompt_:
        while (negative == True) & (n_retry < 6):
            if greedy:
                token_id = torch.argmax(re_distribution).unsqueeze(0)
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