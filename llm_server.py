import requests
import torch
import numpy as np
from llm_router import route
from transformers import AutoModelForCausalLM, AutoTokenizer

token = "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"
API_URL = "https://api-inference.huggingface.co/models/{model_name}/"

parameters = {
  "max_new_tokens": 500,
  #"temperature": 0.01,
  #"top_k": 50,
  #"top_p": 0.95,
  "return_full_text": False
}

headers = {
  'Authorization': f'Bearer {token}',
  'Content-Type': 'application/json'
}

def get_response(query):  
  model_name, local = route(query)
  
  print(f"Chose model {model_name}")

  if local:
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

    # Max number of tokens to generate
    max_new_tokens = 100
    # Context
    generated_ids = inputs["input_ids"]
    logits_history = []

    for _ in range(max_new_tokens):
      with torch.no_grad():
        outputs = model(input_ids=generated_ids)
        logits = outputs.logits

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
    print(f"Logits of chosen tokens: {logits_history}")
    print(f"Mean of logits: {sum(logits_history) / len(logits_history)}")

    # Decode chosen sequence
    response_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

  else:
    # Get large model from HuggingFace API

    payload = {
      "inputs": query,
      "parameters": parameters
    }

    response = requests.post(API_URL.format(model_name=model_name), headers=headers, json=payload)
    response_text = response.json()[0]['generated_text'].strip()

  return response_text
