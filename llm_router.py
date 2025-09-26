import pandas as pd
import numpy as np
import torch
import random
from services.semantic_service import get_similarity_scores, find_k_nearest_neighbors, get_sentence_similarity
from colorama import Fore, Back, Style, init
from transformers import AutoModelForCausalLM, AutoTokenizer, BertTokenizer, BertForSequenceClassification

general_model = "meta-llama/Llama-3.2-3B"
reasoning_model = "Qwen/Qwen2.5-VL-7B-Instruct"
coding_model = "Qwen/Qwen2.5-Coder-3B-Instruct"

available_models = [general_model, reasoning_model, coding_model]

models_and_categories = zip(range(3), available_models)

# check if majority of nearest neighbors are coding-related, otherwise use binary BERT
def knn_router(prompt, k=10):

  print("Routing....")
  # TODO calculate and store the embeddings beforehand, they only need to be calculated once, not on every inference run
  df = pd.read_csv("./data/sample_questions.csv")
  corpus = df.loc[:, "question"].values.tolist()

  # this is a list of row ids with the semantically nearest queries
  neighbors = find_k_nearest_neighbors(prompt, corpus, k)
  row_ids = [entry["corpus_id"] for entry in neighbors[0]]
  print(neighbors)

  # see what is the most frequent model in the nearest neighbors and choose that (majority vote for now)
  neighbor_categories = df.loc[row_ids, "category"]
  print(Fore.RED + f"Neighbor categories: {neighbor_categories.to_list()}" + Style.RESET_ALL)

  # majority vote, if question resembles coding, return the coder model
  majority_category = max(set(neighbor_categories), key=neighbor_categories.count)
  if majority_category == 2:
    return coding_model
  
  # otherwise use a binary classifier
  return bert_router(prompt, "/.models/bert_finetuned_binary/checkpoint-470")
    

# routing strategy based on trained model router
def bert_router(prompt, model_path):
  # load appropriate model
  tokenizer = BertTokenizer.from_pretrained(model_path)
  model = BertForSequenceClassification.from_pretrained(model_path)

  # Tokenize the input
  inputs = tokenizer(prompt, return_tensors="pt")

  # Run inference (no gradients needed)
  with torch.no_grad():
      outputs = model(**inputs)

  # Get predicted class (for classification tasks)
  logits = outputs.logits
  predicted_class = torch.argmax(logits, dim=1)

  return models_and_categories.get(predicted_class)

# random router for benchmarking purposes
def random_router(query):
  return random.choice(available_models)