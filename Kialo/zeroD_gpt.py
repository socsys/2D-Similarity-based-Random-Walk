import pandas as pd
import openai
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score
import sys
from openai import OpenAI
import os

# --- Configuration ---
with open("key.txt", "r") as f:
    my_api_key = f.read().strip()   
client = OpenAI(api_key=my_api_key) 
model = "gpt-4"  # or "gpt-4" if preferred
sample_size=128

for i in range (3):
    # --- Load and Sample Data ---
    df = pd.read_csv("one_K_test_random_walk.csv")
    utterance_col = df.columns[0]      # e.g., 'utterance'
    replying_to_col = df.columns[1]    # e.g., 'parent_utterance'
    label_col = df.columns[-1]         # e.g., 'label'

    # Sample 128 rows to reduce GPT cost
    # Take 10% random sample to reduce GPT call costs
    start=(i+6)*sample_size
    df_sampled = df[start: start+sample_size]

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
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')
    macro_precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
    macro_recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0)

    # Class-wise metrics
    class_precision = precision_score(true_labels, pred_labels, average=None, zero_division=0)
    class_recall = recall_score(true_labels, pred_labels, average=None, zero_division=0)
    class_f1 = f1_score(true_labels, pred_labels, average=None, zero_division=0)



    # --- Print Metrics ---
    print(f"Sample size: {len(df_sampled)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Precision 0: {class_precision[0]:.4f}")
    print(f"Precision 1: {class_precision[1]:.4f}")
    print(f"Recall 0: {class_recall[0]:.4f}")
    print(f"Recall 1: {class_recall[1]:.4f}")
    print(f"f1 0: {class_f1[0]:.4f}")
    print(f"f1 1: {class_f1[1]:.4f}")

    # --- Create a dictionary with metrics as percentages ---
 

    csv_path = "zeroD_Kialo_gpt.csv"

    # --- Metrics dictionary (single row) ---
    metrics_dict = {
        'Sample Size': len(df_sampled),
        'Accuracy (%)': round(accuracy * 100, 2),
        'Macro F1 Score (%)': round(macro_f1 * 100, 2),
        'Precision 0 (%)': round(class_precision[0] * 100, 2),
        'Precision 1 (%)': round(class_precision[1] * 100, 2),
        'Recall 0 (%)': round(class_recall[0] * 100, 2),
        'Recall 1 (%)': round(class_recall[1] * 100, 2),
        'F1 0 (%)': round(class_f1[0] * 100, 2),
        'F1 1 (%)': round(class_f1[1] * 100, 2),
    }

    new_row_df = pd.DataFrame([metrics_dict])

    # ---------------------------------------------------
    # Case 1: File does NOT exist → write header + row
    # ---------------------------------------------------
    if not os.path.isfile(csv_path):
        new_row_df.to_csv(csv_path, index=False)

    # ---------------------------------------------------
    # Case 2: File exists → append row only
    # ---------------------------------------------------
    else:
        new_row_df.to_csv(csv_path, mode='a', index=False, header=False)

        # Read full file to check number of data rows
        full_df = pd.read_csv(csv_path)

        # If exactly 3 experimental runs exist → compute stats
        if len(full_df) == 3:
            mean_row = full_df.mean(numeric_only=True)
            var_row  = full_df.var(numeric_only=True)  # <-- VARIANCE, not std
            std_row = full_df.std(numeric_only=True)

            mean_row['Sample Size'] = 'Mean'
            var_row['Sample Size']  = 'Variance'
            std_row['Sample Size']='Std'

            stats_df = pd.DataFrame([mean_row, var_row, std_row])

            stats_df.to_csv(csv_path, mode='a', index=False, header=False)

    # --- Optional: print to confirm ---
    print("Metrics saved to zeroD_Kialo_gpt.csv as percentages")