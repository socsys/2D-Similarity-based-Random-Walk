import pandas as pd



# import sys
# # --- Load files ---
# df_1d = pd.read_csv("oneD_G_gpt.csv")
# df_common = pd.read_csv("common_2D.csv")

# # --- Extract the required columns as strings ---
# one_d_sent1 = df_1d.iloc[:, 0].astype(str)
# common_sent1 = df_common.iloc[:, 2].astype(str)  # 3rd column is index 2

# # --- Check for missing values ---
# missing = one_d_sent1[~one_d_sent1.isin(common_sent1)]

# # --- Output ---
# if missing.empty:
#     print("✅ All sent1 values from oneD_G_gpt.csv exist in the 3rd column of common_2D.csv")
# else:
#     print(f"❌ {len(missing)} sent1 values are missing from common_2D.csv:")
#     print(missing)
# sys.exit()






# --- Load train and test and combine ---
import pandas as pd

# --- Load train and test and combine ---
df_train = pd.read_csv("GG_train_random_walk.csv")
df_test = pd.read_csv("GG_test_random_walk.csv")
df_combined = pd.concat([df_train, df_test], ignore_index=True)

# --- Load oneD_G_gpt.csv ---
df_1d = pd.read_csv("oneD_G_gpt.csv")

# --- Ensure relevant columns are strings for matching ---
df_combined['sent1'] = df_combined.iloc[:, 2].astype(str)  # sent1 is column index 2
one_d_sent1 = df_1d.iloc[:, 0].astype(str)  # first column of oneD

# --- Collect matched rows ---
matched_rows = []
not_found = []

for idx, value in one_d_sent1.items():
    matched = df_combined[df_combined['sent1'] == value]
    if not matched.empty:
        matched_rows.append(matched.iloc[0])  # Only take the first match
    else:
        not_found.append(value)

# --- Error reporting if any ---
if not_found:
    print(f"❌ {len(not_found)} entries from oneD_G_gpt.csv not found in combined data (sent1 column).")
    for val in not_found:
        print("Missing:", val)
else:
    print("✅ All sent1 values found in combined data.")

# --- Save matched rows to CSV ---
if matched_rows:
    df_common = pd.DataFrame(matched_rows)
    df_common.to_csv("common_2D.csv", index=False)
    print(f"✅ Saved {len(df_common)} matching rows to common_2D.csv")

