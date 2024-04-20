# this code is based on the work from Joon Sung Park (joonspk@stanford.edu)
import json
import random
import openai
import time 
import numpy as np
import pandas as pd
import random
import re

openai.api_key = "xxx"

def temp_sleep(seconds=0.1):
  time.sleep(seconds)

# ============================================================================
# #####################[SECTION 1: CHATGPT-3 Request] ######################
# ============================================================================

def ChatGPT_request(prompt_sys, prompt_user, model="gpt-3.5-turbo"):

  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  # temp_sleep()
  try: 
    completion = openai.ChatCompletion.create(
    model=model,
    messages=[{"role": "user", "content": prompt_user},
    {"role": "system", "content": prompt_sys}],
    temperature=1,
    )
    return completion["choices"][0]["message"]["content"]
  
  except:
    completion = openai.ChatCompletion.create(
          model=model,
          messages=[{"role": "user", "content": prompt_user},
                    {"role": "system", "content": prompt_sys}],
          temperature=1,
      )
    print ("ChatGPT ERROR")
    return "ChatGPT ERROR"

# ============================================================================
# #####################[SECTION 2: Generate prompt] ######################
# ============================================================================


def generate_prompt(curr_input, prompt_lib_file): 
  """
  Takes in the current input (e.g. comment that you want to classifiy) and 
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this 
  function replaces this substr with the actual curr_input to produce the 
  final promopt that will be sent to the GPT3 server. 
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file. 
  RETURNS: 
    a str prompt that will be sent to OpenAI's GPT server.  
  """
  if type(curr_input) == type("string"): 
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):   
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt: 
    prompt_sys = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
    prompt_user = prompt.split("<commentblockmarker>###</commentblockmarker>")[2]
  return prompt_sys.strip(), prompt_user.strip()


def fetch_list(row, round_,  df, mode = "agent"):
    agent, group = row[["agent", "group"]]
    df.t = df.t.astype(int)
    df = df[(df.t >= round_-2) & (df.poem != "ChatGPT ERROR")]
    friend = df[(df.group == group) & (df.agent == agent)].poem.sample(1).tolist()
    enemy = df[df.group != group].poem.sample(1).tolist()
    friend.extend(enemy)
    return friend


# ============================================================================
# #####################[SECTION 3: Response] ######################
# ============================================================================
def clean_response(output):
    splits = split(output.lower(), re.findall('\d\.*', output.lower()))
    poems = []
    notes = []
    for s in splits:
        note = ""
        if "\n\n" in s:
            note = s.split("\n\n")[1]
            if (":" in s.split("\n\n")[0]):
                poems.append(s.split("\n\n")[0].split(":")[1])
            else:
                poems.append(s.split("\n\n")[0])
        else:
            if (":" in s):
                poems.append(s.split(":")[1])
            elif len(s)>0:
                poems.append(s)
        if len(s)>0:
            notes.append(note)
    return poems, notes

def split(txt, seps):
    try:
        default_sep = seps[0]
        # we skip seps[0] because that's the default separator
        for sep in seps[1:]:
            txt = txt.replace(sep, default_sep)
        return [i.strip() for i in txt.split(default_sep)]
    except:
        return [txt]

def clean_v1(text):
    if "Output: " in text:
        text = text.split("Output: ")[1]
    elif ": " in text:
        text = text.split(":")[1]
    text = text.split("2.")[0]
    return text.replace("1.", "")
def process_round(round_, repeat, df, test=True, prompt_lib_file = "prompt_template_v1.txt", mode = "group",
                  model = "gpt-3.5-turbo"):
    response = pd.DataFrame()
    df_ = df[["agent", "group"]].drop_duplicates().reset_index(drop=True)
    print("the number of agents: ", len(df_))
    # update on each agent
    for idx, row in df_.iterrows():
        print(row.tolist(), round_)
        repeat = repeat
        prompt_lst = []
        response_lst = []
        note_lst = []
        for i in range(repeat):
            lst = fetch_list(row, round_, df=df, )
            prompt_sys, prompt_user = generate_prompt(lst, prompt_lib_file)
            prompt_lst.append([prompt_sys, prompt_user])
            temp = ChatGPT_request(prompt_sys, prompt_user, model)
            if temp == "ChatGPT ERROR":
                ChatGPT_request(prompt_sys, prompt_user)
            #poem, note = clean_response(temp)
            response_lst.append(clean_v1(temp))
            #note_lst.extend(note)
        res_df = pd.DataFrame({"poem": response_lst,  "prompt_pair": prompt_lst})  # "note": note_lst,
        res_df["agent"] = row["agent"]
        res_df["group"] = row["group"]
        res_df["t"] = round_

        response = pd.concat([response, res_df])

        if (test) & (len(response) > 3):
            break
        # break
    return response

def process_round_isolated(round_, repeat, df, prompt_lib_file1, prompt_lib_file2, test=True,  mode = "group",
                  model = "gpt-3.5-turbo"):
    response = pd.DataFrame()
    df_ = df[["agent", "group"]].drop_duplicates().reset_index(drop=True)
    #print("the number of agents: ", len(df_))
    # update on each agent
    for idx, row in df_.iterrows():
        repeat = repeat
        prompt_lst1 = []
        prompt_lst2 = []
        response_lst_tmp = []
        response_lst = []
        print(row.tolist(), round_)
        for i in range(repeat):
            lst = fetch_list(row, round_, df=df, )
            friend, foe = lst
            prompt_sys1, prompt_user1 = generate_prompt([friend], prompt_lib_file1)
            prompt_lst1.append([prompt_sys1, prompt_user1])
            temp = ChatGPT_request(prompt_sys1, prompt_user1, model)
            if temp == "ChatGPT ERROR":
                temp = ChatGPT_request(prompt_sys1, prompt_user1)
            #poem, note = clean_response(temp)
            temp = clean_v1(temp)
            response_lst_tmp.append(temp)
            second_lst = [temp, foe]
            prompt_sys2, prompt_user2 = generate_prompt(second_lst, prompt_lib_file2)
            prompt_lst2.append([prompt_sys2, prompt_user2])
            temp = ChatGPT_request(prompt_sys2, prompt_user2, model)
            if temp == "ChatGPT ERROR":
                temp = ChatGPT_request(prompt_sys2, prompt_user2)
            response_lst.append(clean_v1(temp))
        res_df = pd.DataFrame({"poem": response_lst,  "prompt_pair1": prompt_lst1, "prompt_pair2": prompt_lst2, "response_tmp": response_lst_tmp})  # "note": note_lst,
        res_df["agent"] = row["agent"]
        res_df["group"] = row["group"]
        res_df["t"] = round_

        response = pd.concat([response, res_df])

        if (test) & (len(response) > 3):
            break
        # break
    return response