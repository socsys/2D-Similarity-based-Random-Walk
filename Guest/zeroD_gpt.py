import pandas as pd
import openai
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score
import sys
from openai import OpenAI


# --- Configuration ---
my_api_key=""
client = OpenAI(api_key=my_api_key)  # <-- Replace with your actual API key string
model = "gpt-4"  # or "gpt-4" if preferred


# --- Load and Sample Data ---
df = pd.read_csv("one_G_test_random_walk.csv")
utterance_col = df.columns[0]  # First column: utterances
label_col = df.columns[-1]     # Last column: true label

# Take 10% random sample to reduce GPT call costs
df_sampled = df.sample(frac=0.1, random_state=42).copy()

# --- Prediction Loop ---
predictions = []


for utterance in df_sampled[utterance_col]:
    prompt = (
        "Given the following utterance in an online conversation\n"
        'classify the utterance as "1" for "Misogynistic speech" and "0" for "Not misogynistic speech". '
        'Do not return anything else other than "1" or "0".\n'
        f"Utterance: {utterance}"
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
df_sampled.to_csv("zeroD_G_gpt.csv", index=False)

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
metrics_df.to_csv("zeroD_Guest_gpt.csv",  mode='a', index=False, header=False)

# --- Optional: print to confirm ---
print("Metrics saved to zeroD_Guest_gpt.csv as percentages")
