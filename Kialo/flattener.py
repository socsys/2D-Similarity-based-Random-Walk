import pandas as pd
import networkx as nx
import csv
import sys

# Load the CSV
df = pd.read_csv("two_GK_test_random_walk.csv")

# You can adjust these depending on how many comment triples (parent, node, comment) you have
NUM_COMMENTS = 10  # adjust based on your structure
COMMENT_TRIPLE_SIZE = 3
LABEL_INDEX = NUM_COMMENTS * COMMENT_TRIPLE_SIZE

# Prepare output rows
output_rows = []



for idx, row in df.iterrows():

    # Extract label
    label = row[LABEL_INDEX]

    # Extract all comments as triples (parent_id, node_id, comment)
    triples = []
    for i in range(0, NUM_COMMENTS * COMMENT_TRIPLE_SIZE, COMMENT_TRIPLE_SIZE):
        parent = row[i]
        node = row[i+1]
        comment = row[i+2]

        if pd.notna(node) and pd.notna(comment):
            triples.append((str(parent) if pd.notna(parent) else None, str(node), str(comment)))

    # Build a graph to represent the conversation tree
    G = nx.DiGraph()
    id_to_comment = {}

    # Collect all possible IDs from the dataset (to check if parent exists in discussion)
    all_ids_in_discussion = set()
    for i in range(0, NUM_COMMENTS * COMMENT_TRIPLE_SIZE, COMMENT_TRIPLE_SIZE):
        if pd.notna(row[i+1]):
            all_ids_in_discussion.add(str(row[i+1]))

    for parent, node, comment in triples:
        id_to_comment[node] = comment
        if (
            parent and
            parent != node and
            str(parent) in all_ids_in_discussion
        ):
            G.add_edge(parent, node)
        else:
            G.add_node(node)  # add as standalone node if parent is missing or invalid
            # if parent and parent!=node:
            #     G.add_node(parent)

    # Try to find a root (node without incoming edges)
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    visited = set()
    linearized_comments = []

    # Depth-first traversal (pre-order) to linearize
    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        if node in id_to_comment:
            linearized_comments.append(id_to_comment[node])
        for child in G.successors(node):
            dfs(child)

    for root in roots:
        dfs(root)

    if len(roots)==0:
        print('No roots for idx=',idx)
        for node in G.nodes:
            dfs(node)

    if len(linearized_comments)<10:
        for j in range(len(linearized_comments)+1,11):
            linearized_comments.append('')

    # The first comment in the row (the one to classify)
    first_comment = str(row[2]) if pd.notna(row[2]) else ""
    second_comment = str(row[5]) if pd.notna(row[5]) else ""

    # Add row: [first_comment] | [full linearized context as list] | label
    output_rows.append([first_comment, second_comment]+ linearized_comments + [label])


with open("two_LK_test_random_walk.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Utterance','Replying to']+ ['Context_'+str(i) for i in range(1,11)]+ ["Label"])
    for row in output_rows:
        writer.writerow(row)
