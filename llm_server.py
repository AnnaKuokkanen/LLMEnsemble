import requests
from langchain_ollama.llms import OllamaLLM
from llm_router import route

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
    llm = OllamaLLM(model=model_name)

    response_text = llm.invoke(query)

    return response_text

  else:
    payload = {
      "inputs": query,
      "parameters": parameters
    }

    response = requests.post(API_URL.format(model_name=model_name), headers=headers, json=payload)
    response_text = response.json()[0]['generated_text'].strip()

    return response_text
