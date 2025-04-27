# %% [markdown]
# ### LLaMA inference using RAG
# 
# This document will explain the inference using LLaMA by using GPT-4o answers as the RAG.
# 
# For RAG, we will use Chroma database. We will use Chunking of 50 words with overlap of 5 words. But complete sentences will be preserved. For the knowledgebase, the answers from both the training and test data of Gpt-4o-mini will be used.
# 
# For testing, we will only perform the evaluation using the test data

# %%
import os

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import evaluate
import json
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# %% [markdown]
# ### Reading the responses of GPT-4o-mini from test and train data

# %% [markdown]
# Train Data

# %%
ques_list = []
ans_list = []
llama_resp_list = []
gpt_resp_list = []

with open('phase2_data_kabatubare/train_kabatubare.jsonl', 'rb') as file:
    for line in file:
        json_object = json.loads(line)
        ques_list.append(json_object['question'])
        ans_list.append(json_object['answer'])
        llama_resp_list.append(json_object['llama_response_base'])
        gpt_resp_list.append(json_object['gpt_response_base'])
        
train_dataset = pd.DataFrame({'question': ques_list,
                          'answer': ans_list,
                          'llama_response_base': llama_resp_list,
                          'gpt_response_base': gpt_resp_list})
train_dataset

# %% [markdown]
# Test Data

# %%
ques_list = []
ans_list = []
llama_resp_list = []
gpt_resp_list = []

with open('phase2_data_kabatubare/test_kabatubare.jsonl', 'rb') as file:
    for line in file:
        json_object = json.loads(line)
        ques_list.append(json_object['question'])
        ans_list.append(json_object['answer'])
        llama_resp_list.append(json_object['llama_response_base'])
        gpt_resp_list.append(json_object['gpt_response_base'])
        
test_dataset = pd.DataFrame({'question': ques_list,
                          'answer': ans_list,
                          'llama_response_base': llama_resp_list,
                          'gpt_response_base': gpt_resp_list})
test_dataset

# %% [markdown]
# Combining both the test and train data into a single dataframe

# %%
comp_dataset = pd.concat([train_dataset, test_dataset], axis=0).reset_index(drop=True)
comp_dataset

# %% [markdown]
# ### Making Chunks of the Input Data to be used for RAG

# %% [markdown]
# Let's analyze the distribution of characters in each gpt answer

# %%
comp_dataset['gpt_response_base'].apply(lambda x: len(x)).describe()

# %% [markdown]
# Let's use Chunk Size of 1000 and Overlap of 500. This will (on average) ensure that an answer gets divided into 3-4 chunks

# %%
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=500,
    length_function=len,
    is_separator_regex=False,
    separators=["\n\n", "\n", ".", " ",  ""],
)

# %% [markdown]
# Let's chunk the documents and save them into a unified list

# %%
gpt_response_chunks = []

for index, row in comp_dataset.iterrows():
    gpt_response_chunks = gpt_response_chunks + text_splitter.split_text(row['gpt_response_base'])

print(len(gpt_response_chunks))

# %% [markdown]
# Let's generate the ids for each chunk

# %%
gpt_response_chunks_id = list(range(0, len(gpt_response_chunks)))
gpt_response_chunks_id = [str(id) for id in gpt_response_chunks_id]
len(gpt_response_chunks_id)

# %% [markdown]
# ### Setting up the Chroma Database

# %% [markdown]
# Create a persistent client to read the database from a defined directory

# %%
chromadb_client = chromadb.PersistentClient(path="./chroma_db")

# %% [markdown]
# Create or get the `kabatubare_dataset` collection

# %%
kabatubare_dataset_collection = chromadb_client.get_or_create_collection(name="kabatubare_dataset")

# %% [markdown]
# Chroma only allows a batch size of 5461 for addition. Let's prepare the batches with max size of 5000 and put them one by one into the collection

# %%
index_array = np.arange(start=0, stop=len(gpt_response_chunks), step=5000)
index_array = np.append(index_array, len(gpt_response_chunks))
index_array

# %%
for i in range(0, len(index_array)-1):
    start = index_array[i]
    end = index_array[i+1]
    print(f"Adding {start} to {end}")
    kabatubare_dataset_collection.add(
        documents=gpt_response_chunks[start:end],
        ids=gpt_response_chunks_id[start:end],
    )

# %% [markdown]
# Let's see how many embeddings are stored in our collection

# %%
kabatubare_dataset_collection.count()

# %% [markdown]
# 


