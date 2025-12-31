import pandas as pd
import openai
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score
import sys
from openai import OpenAI
import os
import pandas as pd
import numpy as np

# --- Configuration ---
with open("key.txt", "r") as f:
    my_api_key = f.read().strip()   
client = OpenAI(api_key=my_api_key) 
model = "gpt-4"  # or "gpt-4" if preferred
sample_size=128



# --- Load and Sample Data ---
df = pd.read_csv("two_LK_train_random_walk.csv") # or random_LG or 2D_LG...
utterance_col = df.columns[0]      # e.g., 'utterance'
replying_to_col = df.columns[1]    # e.g., 'parent_utterance'
label_col = df.columns[-1]         # e.g., 'label'

# Filter out deleted / removed comments (case-insensitive)
mask = ~df[utterance_col].str.contains(
    "delet|\\[removed\\]",
    case=False,
    na=False
)

df_filtered = df[mask]

# --- Separate by label ---
df_0 = df_filtered[df_filtered[label_col] == 0]
df_1 = df_filtered[df_filtered[label_col] == 1]

# --- Determine the minority class size ---
min_size = min(len(df_0), len(df_1))

# --- Downsample both classes to the same size ---
df_0_balanced = df_0.sample(n=min_size, random_state=42)
df_1_balanced = df_1.sample(n=min_size, random_state=42)

# --- Concatenate back into a single balanced DataFrame ---
df_balanced = pd.concat([df_0_balanced, df_1_balanced]).sample(frac=1, random_state=42)  # shuffle

# --- Take a smaller sample if needed ---
df_sampled = df_balanced[0:400]  # adjust size for GPT calls


# --- Prediction Loop ---
predictions = []

saved = pd.read_csv("saved.csv", header=None)[0]  # Some results are already stored due to multiple runs where previous runs had errors.

i=-1
for _, row in df_sampled.iterrows():
        i+=1
        utterance = row[utterance_col]
        parent = row[replying_to_col]

        if i < 0: # adjust as needed
            result = saved[i]
            print(f"Index {i} - Loaded from file: {result}")
            predictions.append(result)
            continue  # skip the GPT call

        prompt = (
            "Given the following utterance in an online conversation\n"
            'classify the utterance as "1" if it is "Supporting" the utterance it is replying to, '
            'and "0" if it is "Attacking/Opposing" it.\n'
            "IMPORTANT:\n"
            "Return \"-1\" if the utterance is ambiguous or if the intent is unclear.\n"
            "Do not guess. If any doubt exists, return \"-1\".\n"
            'Return only "1", "0", or "-1".\n\n'
            f"Replying to: {parent}\n"
            f"Utterance: {utterance}\n"
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}] }],
                temperature=0
            )
            output = response.choices[0].message.content.strip()
            if output not in ["0", "1", "-1"]:
                print(f"Warning: Unexpected output '{output}', defaulting to '-1'")
                output = "-1"
            else:
                print(output)
        except Exception as e:
            print(f"Error on input '{utterance}': {e}")
            output = "0"  # fallback
            sys.exit()

        predictions.append(output)



# --- Save Updated Data ---
df_sampled["Predicted"] = predictions
df_sampled.to_csv("U_zeroD_G_gpt.csv", index=False)