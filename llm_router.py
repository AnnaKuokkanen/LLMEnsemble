import pandas as pd
import numpy as np
import os.path
from services.semantic_service import get_similarity_scores
from services.data_service import get_training_data
import random
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

  # initialize empty array for storing best model for each query
  optimal_models = np.empty([df.shape[0], 1])

  # for query in df, feed to available models and see which answer is the most semantically similar one to the output
  for i, query in enumerate(df.loc[:, "input"]):
    print(Fore.MAGENTA + f"Completed training {i+1} / {df.shape[0]} queries" + Style.RESET_ALL)
    similarity_scores = []

    for model_name in available_models:
      # do inference with chosen model and the given query
      tokenizer = AutoTokenizer.from_pretrained(model_name)
      model = AutoModelForCausalLM.from_pretrained(model_name)

      model.eval()

      input_ids = tokenizer.encode(query, return_tensors="pt")
      with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=100,
            num_return_sequences=1,
            do_sample=True,  # Enable sampling for more creative results
            #temperature=0.8,  # Controls randomness
            top_k=50,         # Optional: limit sampling to top-k tokens
            #top_p=0.95        # Optional: nucleus sampling
        )

      generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
      similarity_scores.append(get_similarity_scores(generated_text, df.loc[:, "output"]))
    
    # choose the model that outputs the highest similarity score
    optimal_model = available_models[np.argmax(np.array(similarity_scores), axis=None)]
    optimal_models[i] = optimal_model

  # append optimal models into the data frame and write to csv
  df = pd.concat([df, optimal_models], axis=1, ignore_index=True)
  df.to_csv("knn_results.csv", index=False)


# routing strategy based on non-trained router
def knn_router(query, k=1, n=10):
  if not os.path.isfile("knn_results.csv"):
    # if results file does not exist, train the router
    train_knn(n)

  # read previous results
  df = pd.read_csv("knn_results.csv")

  # check if query has already been answered, route to the same model if so 
  matching_row = df[df.loc[:, "query"] == query]

  if not matching_row.empty:
    return matching_row["model"].values

  # list of previous queries to compute semantic similarities against
  prev_queries = df.loc[:, "query"].to_list()
  sem_similarities = get_similarity_scores(prev_queries, query)
  print(Fore.MAGENTA + f"Semantic similarities found: {sem_similarities} \n" + Style.RESET_ALL)

  # find ids of rows with high enough semantic similarity (> 0.7) and return the models 
  model_ids = [i for i in range(len(sem_similarities) - 1) if sem_similarities[i] > 0.7]

  if len(model_ids) == 0:
    return available_models

  # get the models with the confidence scores and return them as possible candidates
  models = df.loc[model_ids, "model"].to_list()

  # backup is queried via API
  """ local = True 
    if model == backup_model:
      local = False """
  
  return models

# routing strategy based on trained model router
def model_router(query, router_model):
  # check if a trained model router exists on device

  # if model does not exist, output
  if router_model == None:
    print("Please provide a trained router model")

# random router for benchmarking purposes
def random_router(query):
  return random.choice(available_models)
