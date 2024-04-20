import os

import pandas as pd
#import names
from utils import *
import numpy as np
import re
from eval_selection import *
import os
import json
import argparse
# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('iteration', type=int,
                    help='the iteration number of experiments')
parser.add_argument('model', type=str,
                    help='GPT model to be used')
args = parser.parse_args()
model = args.model
iteration = args.iteration
print("model to be used ", model)

prompt_lib_file  = "templates/prompt_template_agent_joint.txt"
output_dir = "22_joint_initiation_gpt3"
if not (output_dir in os.listdir()):
    os.mkdir(output_dir)

config = {"output_dir": output_dir, "template": prompt_lib_file}

with open(output_dir + '/config.json', 'w') as fp:
    json.dump(config, fp)

#load dataset
if "t0.csv" in os.listdir(output_dir):
    init_SN = pd.read_csv(output_dir + "/t0.csv")
else:
    init_SN = pd.read_csv("../init_SN.csv")#../init_SN_R2_contrast.csv"),sep = ";"
    init_SN.columns = ["agent", "group", "poem"]
    init_SN = init_SN.groupby("group", as_index=False).sample(2).reset_index()
    init_SN.to_csv(output_dir + '/t0.csv', index=False)
    print("this is a new iteration!")
init_SN.agent = list(range(len(init_SN)))
init_SN = init_SN.reset_index(drop = True)
init_SN["t"] = -1

#init_SN = pd.read_csv(output_dir + "/kids-temp_init_file.csv")
for round_ in range(10):
    output = process_round(round_, 5, df = init_SN, test = False, prompt_lib_file = prompt_lib_file,
                           mode = "group", model = model)
    # ranking outputs
    #output.to_csv(output_dir + "/kids-output_df_" + "t" + str(round_) + ".csv", index= False)
    init_SN = pd.concat([init_SN, output]) #, .merge(output[[str(round_+1),"agent"]], on = "agent")
    #break
    init_SN.to_csv(output_dir + "/temp_init_file" + str(iteration) + ".csv", index = False)

