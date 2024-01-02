from sentence_transformers import SentenceTransformer, LoggingHandler
import logging
import pandas as pd

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

import numpy as np
 
english = pd.read_csv("/homes/rzhang/Multiagents_LLMs/gptlora/dataset/generated_poems/collection_shuffle_True.csv")
doc = np.array_split(english, 3) #.iloc[:30]

#Important, you need to shield your code with if __name__. Otherwise, CUDA runs into issues when spawning new processes.
if __name__ == '__main__':
    n = 0
    for df in doc:
        #Create a large list of 100k sentences
        df = df.reset_index(drop = True)
        sentences = df["stanza"].tolist()
        print(len(sentences), sentences[:2])

        #Define the model
        model = SentenceTransformer('all-mpnet-base-v2')

        #Start the multi-process pool on all available CUDA devices
        pool = model.start_multi_process_pool()

        #Compute the embeddings using the multi-process pool
        emb = model.encode_multi_process(sentences, pool)
        print("Embeddings computed. Shape:", emb.shape)
        
        final = pd.concat([df, pd.DataFrame(emb, index=list(range(len(df))))], axis=1)
        final.iloc[:, :305].to_csv("/homes/rzhang/Multiagents_LLMs/gptlora/dataset/sbert_shuffle_embedding_" + str(n) + ".csv")
        #Optional: Stop the proccesses in the pool
        n = n +1
        print("fold: ", n)
    model.stop_multi_process_pool(pool)