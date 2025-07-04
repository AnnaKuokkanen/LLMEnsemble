import pandas as pd
import numpy as np
import torch
import random
import os.path
from services.semantic_service import get_similarity_scores, find_k_nearest_neighbors
from services.data_service import get_training_data
from colorama import Fore, Back, Style, init
from transformers import AutoModelForCausalLM, AutoTokenizer

math_model = "Qwen/Qwen2-Math-1.5B"
code_model = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
chat_model = "Qwen/Qwen1.5-1.8B-Chat"
instruct_model = "Qwen/Qwen2.5-1.5B-Instruct"
small_model = "Qwen/Qwen3-0.6B"
finnish_model = "TurkuNLP/gpt3-finnish-xl"

#backup_model = "meta-llama/Llama-3.3-70B-Instruct"

available_models = [math_model, code_model, small_model]

# method for finding best model for each query via semantic similarity of replies
def train_knn(n):
  df = get_training_data(n)

  # add placeholder columns for model name and similarity score
  optimal_models = pd.DataFrame(np.empty((df.shape[0], 1)))
  similarity_scores = pd.DataFrame(np.full((df.shape[0], 1), -999999))

  df = pd.concat([df, optimal_models, similarity_scores], axis=1, ignore_index=True).rename(columns={0: "input", 1: "output", 2: "model_name", 3: "similarity_score"})

  # for query in df, feed to available models and see which answer is the most semantically similar one to the output
  for model_name in available_models:
    print(Fore.MAGENTA + f"Next testing model {model_name}" + Style.RESET_ALL)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.eos_token_id is None:
      tokenizer.eos_token = ""
    eos_token_id = tokenizer.eos_token_id

    model.eval()

    # do inference with chosen model for each query
    for i, query in enumerate(df.loc[:, "input"]):
      input_ids = tokenizer.encode(query, return_tensors="pt")
      with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            #max_length=100,
            num_return_sequences=1,
            do_sample=True,  # Enable sampling for more creative results
            #temperature=0.8,  # Controls randomness
            #top_k=50,         # Optional: limit sampling to top-k tokens
            #top_p=0.95        # Optional: nucleus sampling
        )

      generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

      # get similarity score between generated text and ground truth
      similarity_score = get_sentence_similarity(generated_text, df.loc[i, "output"])
      print(similarity_score)
      print(df.loc[i, "similarity_score"])

      # if similarity score is better, update current model to
      if similarity_score > df.loc[i, "similarity_score"]:
        df.loc[i, "model_name"] = model_name
        df.loc[i, "similarity_score"] = similarity_score

      print(f"Completed training {i+1} / {df.shape[0]} queries")

    # delete model from memory
    del model

  # write results to csv
  df.to_csv("knn_results.csv", index=False)


# routing strategy based on non-trained router
def knn_router(query, k=1, n=10):
  if not os.path.isfile("data/knn_results.csv"):
    # if results file does not exist, train the router to obtain n data points
    train_knn(n)

  # read previous results
  df = pd.read_csv("data/knn_results.csv")

  # next we need to find k nearest neighbors in the embedding space in the training examples
  # TODO find a way to store the embeddings, they only need to be calculated once, not on every inference run
  corpus = df.loc[:, "input"].values.tolist()

  # this is a list of row ids with the semantically nearest queries
  knn_results = find_k_nearest_neighbors(query, corpus, k)
  row_ids = [entry["corpus_id"] for entry in knn_results[0]]
  print(knn_results)

  # see what is the most frequent model in the nearest neighbors and choose that (majority vote for now)
  model_candidates = df.loc[row_ids, "model_name"]
  print(Fore.RED + f"Model candidates: {model_candidates.to_list()}" + Style.RESET_ALL)

  # majority vote
  best_model = max(set(model_candidates), key=model_candidates.count)
  return best_model

# random router for benchmarking purposes
def random_router(query):
  return random.choice(available_models)


# routing strategy based on trained model router
def model_router(query, router_model):
  # check if a trained model router exists on device

  # if model does not exist, output
  if router_model == None:
    print("Please provide a trained router model")