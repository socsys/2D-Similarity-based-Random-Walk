import csv

def meaningful_length(row):
    """
    Count meaningful feature values (ignore last column = label).
    Empty values count if followed by a non-empty value.
    Do not count trailing empty values.
    """
    features = row[:-1]  # all columns except label
    last_non_empty = -1

    for i, v in enumerate(features):
        if v.strip() != "":
            last_non_empty = i

    return last_non_empty + 1 if last_non_empty != -1 else 0


def trim_rows(file1_path, file2_path, out1_path, out2_path):
    # Read both CSVs
    with open(file1_path, newline='', encoding='utf-8') as f1, \
         open(file2_path, newline='', encoding='utf-8') as f2:

        r1 = list(csv.reader(f1))
        r2 = list(csv.reader(f2))

    headers1 = r1[0]
    headers2 = r2[0]

    if headers1 != headers2:
        raise ValueError("Headers do not match")

    cleaned1 = [headers1]
    cleaned2 = [headers1]

    for row1, row2 in zip(r1[1:], r2[1:]):

        len1 = meaningful_length(row1)
        len2 = meaningful_length(row2)
        min_len = min(len1, len2)

        # Prepare output rows of the same length as input
        new_row1 = row1[:]  # copy
        new_row2 = row2[:]  # copy

        # Replace trimmed features with empty strings (preserve label position)
        for i in range(min_len, len(row1) - 1):  # -1 ensures we don't touch label
            new_row1[i] = ""
            new_row2[i] = ""

        cleaned1.append(new_row1)
        cleaned2.append(new_row2)

    # Save
    with open(out1_path, "w", newline='', encoding="utf-8") as f1, \
         open(out2_path, "w", newline='', encoding="utf-8") as f2:

        csv.writer(f1).writerows(cleaned1)
        csv.writer(f2).writerows(cleaned2)


# ------------------ EXAMPLE USAGE ------------------
trim_rows(
    "one_LK_test_random_walk.csv",
    "two_LK_test_random_walk.csv",
    "Equalized_one_LK_test_random_walk.csv",
    "Equalized_two_LK_test_random_walk.csv"
)
