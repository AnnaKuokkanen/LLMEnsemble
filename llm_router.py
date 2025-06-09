import random

math_model = "Qwen/Qwen2-Math-1.5B"
code_model = "qQwen/Qwen2.5-Coder-1.5B-Instruct"
chat_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

backup_model = "meta-llama/Llama-3.3-70B-Instruct"

available_models = [math_model, code_model,chat_model, backup_model]

def route(query):
    # naive router, just pick a random model

    #model = random.choice(available_models)
    model = math_model
    local = True

    # backup is queried via API
    if model == backup_model:
        local = False
    
    return model, local