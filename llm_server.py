import os.path
import requests
import torch
import time
import numpy as np
import pandas as pd
from csv import writer, reader
from llm_router import knn_router
from transformers import AutoModelForCausalLM, AutoTokenizer
from colorama import Fore, Back, Style, init

token = "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"
API_URL = "https://api-inference.huggingface.co/models/{model_name}/"

parameters = {
  "max_new_tokens": 300,
  "temperature": 0.01,
  "return_full_text": False
}

headers = {
  'Authorization': f'Bearer {token}',
  'Content-Type': 'application/json'
}

def initialize(path="results.csv"):
  # create a csv file for storing results 
  if not os.path.isfile(path):
    with open("results.csv", "w") as f:
      writer_object = writer(f)
      writer_object.writerow(["query", "model", "logit_1", "logit_2", "logit_3", "mean_logits", "tokenized_length", "latency", "response"])
      f.close()

def record_results(query, model_name, logits, mean_logits, token_representation_length, response, latency):
  # store query, the model that was chosen and some metrics for future reference

  # Check if the same query has been answered before 
  df = pd.read_csv("results.csv")
  matching_row = df[df["query"] == query]
  
  # if has not ben answered or we are not as confident in the answer, record
  if matching_row.empty:
    with open("results.csv", "a") as f:
      writer_object = writer(f)
      # if the same query has been answered, record results
      writer_object.writerow([query, model_name, logits[0], logits[1], logits[2], mean_logits, token_representation_length, latency, response])
      f.close()
  elif np.mean(matching_row.iloc[0, :]['logit_1':'logit_3'].values) < sum(logits[:3])/3 and matching_row.iloc[0, :]['tokenized_length'] >= token_representation_length:
    # replace row with updated information
    index = matching_row.index[0]
    df.iloc[index, :] = {"query": query, "model": model_name, "logit_1": logits[0], "logit_2": logits[1], "logit_3": logits[2], "mean_logits": mean_logits, "tokenized_length": token_representation_length, "latency": latency, "response": response}
    df.to_csv("results.csv", index=False)

def get_response(query, temperature=1.0):
  # get the potential models
  initialize()
  potential_models = knn_router(query, k=3, n=10)
  print(f"Potential models: {potential_models} \n")

  # start measuring latency of generation process
  start = time.time()

  # if multiple potential models, generate answers with all of them and record the best one 
  for model_name in potential_models:
    print(f"Chose model {model_name} \n")
    
    # Get pretrained model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Determine EOS token 
    if tokenizer.eos_token_id is None:
      tokenizer.eos_token = ""
    eos_token_id = tokenizer.eos_token_id

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize user prompt
    inputs = tokenizer(query, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"][0]  # Remove batch dimension

    # Convert token IDs to token strings
    token_strings = [tokenizer.decode([token_id]) for token_id in input_ids]
    print(Fore.CYAN + f"Tokens as plain text: {token_strings} \n" + Style.RESET_ALL)

    # Max number of tokens to generate
    max_new_tokens = 100
    # Context
    generated_ids = inputs["input_ids"]
    logits_history = []

    for _ in range(max_new_tokens):
      with torch.no_grad():
        outputs = model(input_ids=generated_ids)
        logits = outputs.logits
        logits /= temperature

      # Get the logits for the next token (last position)
      next_token_logits = logits[:, -1, :]  # shape: (batch_size, vocab_size)
      logits_history.append(torch.max(next_token_logits).item())

      # Choose next token with max logit score
      next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

      # Exit if the end of a sequence
      if next_token_id.item() == eos_token_id:
        break

      # Append next token to input for next step
      generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

    # Print information about logits
    print(Fore.CYAN + Style.BRIGHT + f"Logits of chosen tokens: {logits_history} \n" + Style.RESET_ALL)
    mean_logits = sum(logits_history) / len(logits_history)
    print(f"Mean of logits: {mean_logits} \n")

    # Decode chosen sequence
    response_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(Fore.BLUE + Style.BRIGHT + f"Response by {model_name}: {response_text} \n" + Style.RESET_ALL)

    # stop measuring latency of generation process
    end = time.time()
    latency = (end - start) * 1000
    # write results 
    record_results(query, model_name, logits_history, mean_logits, len(input_ids), response_text, latency)
    
  df = pd.read_csv("results.csv")
  response = df[df.loc[:, "query"] == query]["response"].values[0]
  return response

  """ else:
    # Get large model from HuggingFace API

    payload = {
      "inputs": query,
      "parameters": parameters
    }

    response = requests.post(API_URL.format(model_name=model_name), headers=headers, json=payload)
    response_text = response.json()[0]['generated_text'].strip() """
