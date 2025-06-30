import pandas as pd

# Orca math dataset, 200K
df_math = pd.read_parquet("hf://datasets/microsoft/orca-math-word-problems-200k/data/train-00000-of-00001.parquet")

# Open critic GPT code dataset, 55.1K
df_code = pd.read_json("hf://datasets/Vezora/Open-Critic-GPT/Open-Critic-GPT.jsonl", lines=True)

# tulu personal instructions, 30K
#df_instruct = pd.read_parquet("hf://datasets/allenai/tulu-3-sft-personas-instruction-following/data/train-00000-of-00001.parquet")

# OpenScience, 75.7K
df_science = pd.read_json("hf://datasets/nvidia/OpenScience/OS-Q3-235B-4.jsonl", lines=True)

# method that selects n random rows from each data set and combines them into a single data set
def get_training_data(n):
    # get n rows from each data set
    math = df_math.sample(n=n).rename(columns={"question": "input", "answer": "output"}, errors="raise")
    code = df_code.sample(n=n).iloc[:, 1:].rename(columns={"Human": "input", "Assistant": "output"}, errors="raise")
    science = df_science.sample(n=n)
    #df_instruct = df_instruct.sample(n=n)

    # combine the rows and return the data frame
    return pd.concat([math, code, science], axis=0, ignore_index=True)