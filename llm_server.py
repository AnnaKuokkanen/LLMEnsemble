import requests
from sys import argv
import torch
import numpy as np
import pandas as pd
from csv import writer, reader
from llm_router import knn_router, bert_router
from transformers import AutoModelForCausalLM, AutoTokenizer

# Function call to router based on routing strategy
def llm_router(prompt, routing_strat="inc_knn"):
  # RAG to add context

  # determine if prompt is relevant by searching the RAG base and/or comparing it to sample questions
  """ if min_distance > hyperparameter:
        return "I apologize, the question does not seem relevant to our operations"
      
      if top_docs == None:
        return "I could not find recent information on this" """

  # two possible router options for now
  model_name  = knn_router(prompt) if routing_strat == "inc_knn" else bert_router(prompt, "/.models/bert_finetuned_multi/checkpoint-309")
  return generate(prompt, model_name)

def generate(prompt, model_name):
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(model_name)

  tokenizer.pad_token_id = tokenizer.eos_token_id

  # Move model to GPU if available
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  
  model.eval()

  input_ids = tokenizer.encode(prompt, return_tensors="pt")

  output_ids = model.generate(
      **input_ids,
      max_length=100,
      num_return_sequences=1,
      do_sample=True,
      temperature=0.8,
      #top_k=50,
      #top_p=0.95
  )

  del model

  answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
  return answer


def main():
    llm_router(argv[0])

if __name__ == "__main__":
    main()