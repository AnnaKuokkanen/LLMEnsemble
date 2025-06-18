import pandas as pd
import os.path
from semantic_service import get_similarity_scores

math_model = "Qwen/Qwen2-Math-1.5B"
code_model = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
chat_model = "Qwen/Qwen1.5-1.8B-Chat"
instruct_model = "Qwen/Qwen2.5-1.5B-Instruct"
finnish_model = "TurkuNLP/gpt3-finnish-xl"

#backup_model = "meta-llama/Llama-3.3-70B-Instruct"

available_models = [math_model, finnish_model, code_model, chat_model, instruct_model]

# method returns either 
def route(query):
  if not os.path.isfile("results.csv"):
    # if results file does not exist, send query to all models
    return available_models

  # read previous results
  df = pd.read_csv("results.csv")

  # check if query has already been answered, route to the same model if so 
  matching_row = df[df.loc[:, "query"] == query]

  if not matching_row.empty:
    return matching_row["model"].values

  # list of previous queries to compute semantic similarities against
  prev_queries = df.loc[:, "query"].to_list()
  sem_similarities = get_similarity_scores(prev_queries, query)
  print(sem_similarities)

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

""" def compute_reward():
  # function for computing a reward
  for model in available_models:
    # get data on model's performance

  return 0 """