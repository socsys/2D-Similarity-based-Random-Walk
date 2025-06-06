import pandas as pd

# Step 1: Load all datasets
one_train = pd.read_csv("one_GK_train_random_walk.csv")
one_test = pd.read_csv("one_GK_test_random_walk.csv")
full_train = pd.read_csv("GK_train_random_walk.csv")
full_test = pd.read_csv("GK_test_random_walk.csv")

# Combine train and test into one full dataset for matching
full_data = pd.concat([full_train, full_test], ignore_index=True)

# Helper function to match and replace rows
def replace_with_full(one_df, full_df):
    replaced_rows = []
    for idx, row in one_df.iterrows():
        #s1 = row['sent1']
        id1 = row['id1']
        parent_id1=row['parent_id1']

        # Look for the same sent1 and sent2 in full data
        match = full_df[(full_df['id1'] == id1) & (full_df['parent_id1'] == parent_id1)]
        if not match.empty:
            replaced_rows.append(match.iloc[0])
        else:
            raise ValueError(f"No match found for id1: {id1}")
    
    return pd.DataFrame(replaced_rows)

# Step 2: Replace rows using full data (train + test)
two_train = replace_with_full(one_train, full_data)
two_test = replace_with_full(one_test, full_data)

# Step 3: Save updated structured versions
two_train.to_csv("two_GK_train_random_walk.csv", index=False)
two_test.to_csv("two_GK_test_random_walk.csv", index=False)

# Step 4: Drop 'id' and 'parent_id' prefixed columns for unstructured versions
def drop_id_columns(df):
    return df[[col for col in df.columns if not col.startswith('id') and not col.startswith('parent_id')]]

two_train_unstructured = drop_id_columns(two_train)
two_test_unstructured = drop_id_columns(two_test)

# Save unstructured versions
two_train_unstructured.to_csv("two_K_train_random_walk.csv", index=False)
two_test_unstructured.to_csv("two_K_test_random_walk.csv", index=False)

print("✅ Done. Files saved:")
print("- two_GK_train_random_walk.csv")
print("- two_GK_test_random_walk.csv")
print("- two_K_train_random_walk.csv")
print("- two_K_test_random_walk.csv")
