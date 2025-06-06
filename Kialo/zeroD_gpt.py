import pandas as pd
import openai
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score
import sys
from openai import OpenAI
import os

# --- Configuration ---
my_api_key = ""  # Replace with your OpenAI API key
client = OpenAI(api_key=my_api_key)  # <-- Replace with your actual API key string
model = "gpt-4"  # or "gpt-4" if preferred


# --- Load and Sample Data ---
df = pd.read_csv("one_K_test_random_walk.csv")
utterance_col = df.columns[0]      # e.g., 'utterance'
replying_to_col = df.columns[1]    # e.g., 'parent_utterance'
label_col = df.columns[-1]         # e.g., 'label'

# Sample 128 rows to reduce GPT cost
df_sampled = df.sample(n=128, random_state=23).copy()

saved = pd.read_csv("saved.csv", header=None)[0]  # Some results are already stored due to multiple runs where previous runs had errors.

# Predictions list
predictions = []
i=-1
# Loop over the sampled DataFrame rows
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
        'Return only "1" or "0".\n\n'
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
        if output not in ["0", "1"]:
            print(f"Warning: Unexpected output '{output}', defaulting to '0'")
            output = "0"
            sys.exit()
        else:
            print(output)
    except Exception as e:
        print(f"Error on input '{utterance}': {e}")
        output = "0"  # fallback
        sys.exit()

    predictions.append(output)

# --- Save Updated Data ---
df_sampled["Predicted"] = predictions
df_sampled.to_csv("zeroD_K_gpt.csv", index=False)

# --- Evaluation ---
true_labels = df_sampled[label_col].astype(int)
pred_labels = df_sampled["Predicted"].astype(int)

accuracy = accuracy_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels, average="macro")
precision = precision_score(true_labels, pred_labels, average="macro")
recall = recall_score(true_labels, pred_labels, average="macro")

# try:
#     pr_auc = average_precision_score(true_labels, pred_labels)
# except:
#     pr_auc = "Not computable (requires probabilities)"

# --- Print Metrics ---
print(f"Sample size: {len(df_sampled)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1 Score: {f1:.4f}")
print(f"Macro Precision: {precision:.4f}")
print(f"Macro Recall: {recall:.4f}")
#print(f"PR AUC: {pr_auc}")


# --- Create a dictionary with metrics as percentages ---
metrics_dict = {
    'Sample Size': [len(df_sampled)],
    'Accuracy (%)': [round(accuracy * 100, 2)],
    'Macro F1 Score (%)': [round(f1 * 100, 2)],
    'Macro Precision (%)': [round(precision * 100, 2)],
    'Macro Recall (%)': [round(recall * 100, 2)]
 #  'PR AUC (%)': [round(pr_auc * 100, 2)]
}

# --- Convert to DataFrame and save to CSV ---
metrics_df = pd.DataFrame(metrics_dict)

file_path = "zeroD_Kialo_gpt.csv"
# Check if file exists
file_exists = os.path.exists(file_path)
# Write to CSV
metrics_df.to_csv(file_path, mode='a', index=False, header=not file_exists)


# --- Check if the file has 3 rows now ---
existing_df = pd.read_csv(file_path)

if len(existing_df) == 3:
    # Exclude the first column from mean calculation
    numeric_cols = existing_df.columns[1:]
    mean_values = existing_df[numeric_cols].mean().round(2)

    # Create the mean row with 'mean' as first column
    mean_row = pd.Series(['mean'] + mean_values.tolist(), index=existing_df.columns)

    # Append mean row
    mean_df = pd.DataFrame([mean_row])
    mean_df.to_csv(file_path, mode='a', index=False, header=False)



# --- Optional: print to confirm ---
print("Metrics saved to zeroD_Kialo_gpt.csv as percentages")
