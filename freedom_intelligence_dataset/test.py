# %% [markdown]
# ### LLaMA Supervised Fine-Tuning
# 
# This document will perform the inference on evaluation dataset of freedom intelligence using the base model

# %%
import os

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# %%
import pandas as pd
import json
import torch
import pickle
from unsloth import FastLanguageModel
from datasets import Dataset
from tqdm  import tqdm
import evaluate

# %% [markdown]
# #### Reading the Question and Answer Pairs from Test Dataset Phase 2

# %%
ques_list = []
ans_list = []
llama_resp_list = []
gpt_resp_list = []

with open('phase2_data_freedom_intelligence/test_freedom_intelligence.jsonl', 'rb') as file:
    for line in file:
        json_object = json.loads(line)
        ques_list.append(json_object['question'])
        ans_list.append(json_object['answer'])

# %%
test_dataset = pd.DataFrame({'question': ques_list,
                          'answer': ans_list})
test_dataset

# %% [markdown]
# ### Inference

# %%
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length = 1024, # [NEW!] Max sequence length for the model
    load_in_4bit = False, # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = True, # [NEW!] We have full finetuning now!
    dtype=None, #None for auto-detection. Can be torch.bfloat16 or torch.float16 (will be automatically detected)
    device_map="auto"
)

# %% [markdown]
# Implementing sample-by-sample inference. (Batch Inference doesn't work well for fine-tuned model adapters as responses like `P P P P` are being produced)

# %%
def get_llama_response(question_input: str):
    
    llama_input = [{"role": "system", "content": "You are a medical knowledge assistant trained to provide information and guidance on various health-related topics."},
                    {"role": "user", "content": question_input}]

    prompt = tokenizer.apply_chat_template(llama_input, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt").to(model.device)
    temp_resp = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=1024,
        num_return_sequences=1
    )

    resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
    resp = resp[len(temp_resp):] #getting only the response part (i.e., assistant)
    
    return resp

# %%
# Implementing the Unsloth Fast Inference
FastLanguageModel.for_inference(model)

llama_responses_base = []
for index, row in tqdm(test_dataset.iterrows(), total=len(test_dataset)):
    question_input = row['question']
    llama_resp = get_llama_response(question_input)
    llama_responses_base.append(llama_resp)

with open('phase2_freedom_intelligence/llama_responses_base.pkl', 'wb') as file:
    pickle.dump(llama_responses_base, file)

# %%
with open('phase2_freedom_intelligence/llama_responses_base.pkl', 'rb') as file:
    llama_responses_base = pickle.load(file)

# %% [markdown]
# ### Saving the LLaMA Responses into the complete dataframe

# %%
test_dataset['llama_responses_base'] = llama_responses_base
test_dataset

# %% [markdown]
# ### Calculating the BLEU Results for Phase 3

# %% [markdown]
# LLaMA Response Groundtruth Fine-Tuned

# %%
bleu_eval = evaluate.load("bleu")
bleu_results = bleu_eval.compute(predictions=test_dataset['llama_responses_base'].to_list(), references=test_dataset['answer'].to_list())
bleu_results

# %% [markdown]
# ### Calculating the ROUGE Results for Phase 3

# %% [markdown]
# LLaMA Response Fine-Tuned

# %%
rouge_eval = evaluate.load("rouge")
rouge_results = rouge_eval.compute(predictions=test_dataset['llama_responses_base'].to_list(), references=test_dataset['answer'].to_list())
rouge_results

# %%



