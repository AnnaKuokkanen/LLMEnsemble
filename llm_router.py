import random


math_model = "qwen2-math:1.5b"
code_model = "qwen2.5-coder:0.5b"
chat_model = "deepseek-r1:1.5b"

backup_model = "meta-llama/Llama-3.3-70B-Instruct"

available_models = [math_model, code_model,chat_model, backup_model]

def route(query):
    # naive router, just pick a random model

    model = random.choice(available_models)
    local = True

    # backup is queried via API
    if model == backup_model:
        local = False
    
    return model, local