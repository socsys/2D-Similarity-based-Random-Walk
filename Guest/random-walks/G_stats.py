import pickle as pkl
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import networkx as nx

import os
import csv
import pickle as pkl
import pandas as pd
import numpy as np
import random
import math
from sentence_transformers import SentenceTransformer, util
import sys
import networkx as nx
from matplotlib import pyplot as plt
import seaborn as sns
import itertools
from collections import defaultdict
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
import textwrap
import matplotlib.lines as mlines

walk_length=10


label_map={0:'Non-Hate',1:'Hate'}

# Global dictionary to store longest paths for each label category
longest_paths = {
    0: [],
    1: [],
}
mean_branching_factors = {
    0: [],
    1: [],
}

relevance_scores = {
    0: [],
    1: [],
}

relevance_scores1 = {
    0: [],
    1: [],
}

longest_paths_all=[]
mean_branching_factors_all=[]


def sbert_cosine_similarity(node, nbd_sentences, num_candidates):
    try:
        #print(node[:],' AGAINST: ', [n[:] for n in nbd_sentences] )
        global model
        nbd_sent_embeddings = model.encode(nbd_sentences, convert_to_tensor=True)
        node_embedding = model.encode(node, convert_to_tensor=True)
        hits = util.semantic_search(node_embedding, nbd_sent_embeddings, top_k=num_candidates)
        return hits[0]
    except Exception as e: 
        print(node,' AGAINST Error: ', [n for n in nbd_sentences] )
        print(f"Error occurred: {e}")  # Print the exception message
        sys.exit()




def random_graph_walk(sentences, data, node_id, child_edges, walk_len):

    print('generating new 2D graph walk')

    #print('Main Comment:', [node_id,data['node'][node_id]['text']] )
    #print('Category:',data['node'][node_id]['label'])

    #some cleaning first to replace all nans
    for n_id in data['node']:
        for key, value in data['node'][n_id].items():
            if key=='text' and pd.isna(value):  # Check for NaN
                data['node'][n_id][key] = ''  # Replace NaN with empty string

    edge= data['edge'][node_id]
    if len(edge.keys()) > 0:
        sentences[0] = list(edge.keys())[0] #parent id
    else:
        sentences[0]=''
    sentences[1] = node_id # node id
    sentences[2] = data['node'][node_id]['text']#node text  
    label = data['node'][node_id]['label']
    chosen_node_ids = [node_id]
    visited_nodes=[node_id]
    indx = 3
    retries=0
    original_node_id=node_id
    scores=[] #relevance scores

    extended_edges=[] # we do not extend same edge twice

    # Adding parent mandatorily
    edge = data['edge'][node_id]
    if len(edge.keys()) > 0:
        child_id=node_id
        node_id = list(edge.keys())[0] # there is only one parent
        if node_id not in data['node'] and node_id not in data['edge']:
            return sentences, label, chosen_node_ids, extended_edges,0 #relevance score is 0
        else:
            e=data['edge'][node_id]
            if len(e.keys()) > 0:
                sentences[3] = list(e.keys())[0]
            else:
                sentences[3] = ''
            sentences[4] = node_id
            sentences[5] = data['node'][node_id]['text'] 
            chosen_node_ids.append(node_id)
            extended_edges.append([child_id, node_id]) # save to extended edges
            indx += 3
    ########
    choices = []
    choices_text = []
 
    while indx < walk_len*3: 

        visited_nodes.append(node_id)
        edge = data['edge'][node_id]

        #parent node
        #not list(edge.keys())[0] in visited_nodes
        if len(edge.keys()) > 0 and list(edge.keys())[0] in data['node'].keys(): 
            parent_id=list(edge.keys())[0]
            if not parent_id in choices:
                choices.append(parent_id) #append parent
                choices_text.append([parent_id,data['node'][parent_id]['text']])
                if not [node_id,parent_id] in extended_edges and not [parent_id,node_id] in extended_edges:
                    extended_edges.append([node_id,parent_id]) # save to extended edges

        if node_id in child_edges:
            for child_id in child_edges[node_id]: #there are possibly multiple children
                #not child_id in visited_nodes and
                if  child_id in data['node'].keys(): 
                    if not child_id in choices:
                        choices.append(child_id) #append child
                        choices_text.append([child_id,data['node'][child_id]['text']])
                        if not [node_id,child_id] in extended_edges and not [child_id,node_id] in extended_edges:
                                extended_edges.append([node_id,child_id]) # save to extended edges
      


        if len(choices) == 0:
            return sentences, label, chosen_node_ids, extended_edges, sum(scores)/len(scores) if scores else 0
        
        hits = sbert_cosine_similarity(
            data['node'][original_node_id]['text'], [t[1] for t in choices_text], len(choices))
        
        # print('************** 2D *****************************')

        # print('choices:',choices)
        # print('hits:',hits)

        # Extract probabilities from similarity scores
        probs = [abs(hit['score']) for hit in hits]
        total = np.sum(probs)
        probs = probs / total if total > 0 else np.full_like(probs, 1 / len(probs))

        #print('probs=',probs)

        # Select a node based on probabilities
        selected_hit = random.choices(hits, weights=probs)[0]
        selected_corpus_id = selected_hit['corpus_id']
        node = choices[selected_corpus_id]  # Map back to the correct node ID
        score = selected_hit['score']  # Retrieve the actual score for the selected node

        # print('selected node:',node)
        # print('score',score)

        # print('************** 2D *****************************')


        if node not in chosen_node_ids and not data['node'][node]['text']=='' :
            e=data['edge'][node]
            if len(e.keys()) > 0:
                sentences[indx] = list(e.keys())[0]
            else:
                sentences[indx] = ''
            sentences[indx+1]= node #child
            sentences[indx+2] = data['node'][node]['text'] #text
            chosen_node_ids.append(node)
            scores.append(score)
            indx += 3
            retries=0
        else:
            retries+=1
            if retries>walk_len:
                break
 
        node_id = node
        choices.remove(node_id)
        choices_text.remove([node_id,data['node'][node_id]['text']])
        print('length of choices=',len(choices))
        #debugging
        # print('**********debugging***********************')
        #print('selected choice', [node_id,data['node'][node_id]['text']] )
        #print('indx=',indx,', choices=',choices)
        # print('choices_text=',choices_text) 
        # print('******************************************')

        
    return sentences, label, chosen_node_ids, extended_edges,sum(scores)/len(scores) if scores else 0

def one_d_random_graph_walk(sentences, data, node_id, child_edges, walk_len):

    print('generating new 1D graph walk')

    #print('Main Comment:', [node_id,data['node'][node_id]['text']] )
    #print('Category:',data['node'][node_id]['label'])

    #some cleaning first to replace all nans
    for n_id in data['node']:
        for key, value in data['node'][n_id].items():
            if key=='text' and pd.isna(value):  # Check for NaN
                data['node'][n_id][key] = ''  # Replace NaN with empty string

    edge= data['edge'][node_id]
    if len(edge.keys()) > 0:
        sentences[0] = list(edge.keys())[0] #parent id
    else:
        sentences[0]=''
    sentences[1] = node_id
    sentences[2] = data['node'][node_id]['text']    
    label = data['node'][node_id]['label']
    chosen_node_ids = [node_id]
    indx = 3
    retries = 0
    original_node_id=node_id
    scores=[] #relevance scores

    extended_edges=[] # we do not extend same edge twice

    # Adding parent mandatorily
    edge = data['edge'][node_id]
    if len(edge.keys()) > 0:
        child_id=node_id
        node_id = list(edge.keys())[0] # there is only one parent
        if node_id not in data['node'] and node_id not in data['edge']:
            return sentences, label, chosen_node_ids, extended_edges,0 #relevance score is 0
        else:
            e=data['edge'][node_id]
            if len(e.keys()) > 0:
                sentences[3] = list(e.keys())[0]
            else:
                sentences[3] = ''
            sentences[4] = node_id
            sentences[5] = data['node'][node_id]['text']
            chosen_node_ids.append(node_id)
            extended_edges.append([child_id, node_id]) # save to extended edges
            indx += 3
    ########

    while indx < walk_len*3: 

        choices = []
        choices_text = []

        #visited_nodes.append(node_id)
        edge = data['edge'][node_id]

        #parent node
        if len(edge.keys()) > 0 and list(edge.keys())[0] in data['node'].keys(): 
            parent_id=list(edge.keys())[0]
            choices.append(parent_id) #append parent
            choices_text.append([parent_id,data['node'][parent_id]['text']])
            if [node_id,parent_id] not in extended_edges and [parent_id,node_id] not in extended_edges:
                extended_edges.append([node_id,parent_id]) # save to extended edges

        if node_id in child_edges:
            for child_id in child_edges[node_id]: #there are possibly multiple children
                if child_id in data['node'].keys(): 
                    choices.append(child_id) #append child
                    choices_text.append([child_id,data['node'][child_id]['text']])
                    if [node_id,child_id] not in extended_edges and [child_id,node_id] not in extended_edges:
                        extended_edges.append([node_id,child_id]) # save to extended edges
      
        print('length of choices=',len(choices))

        if len(choices) == 0:
            return sentences, label, chosen_node_ids, extended_edges,sum(scores)/len(scores) if scores else 0

        
        hits = sbert_cosine_similarity(
            data['node'][node_id]['text'], [t[1] for t in choices_text], len(choices))
        
        probs = [abs(hit['score']) for hit in hits]
        total = np.sum(probs)
        probs = probs / total if total > 0 else np.full_like(probs, 1 / len(probs))

        #print('probs=',probs)

        # Select a node based on probabilities
        selected_hit = random.choices(hits, weights=probs)[0]
        selected_corpus_id = selected_hit['corpus_id']
        node = choices[selected_corpus_id]  # Map back to the correct node ID

        #relevance to original node
        original_node_relevance=sbert_cosine_similarity(
            data['node'][original_node_id]['text'], [choices_text[selected_corpus_id][1]], 1)
        score=original_node_relevance[0]['score']

        # print('selected node:',node)
        # print('score',score)

        # print('************** 1D *****************************')


        if node not in chosen_node_ids:
            e=data['edge'][node]
            if len(e.keys()) > 0:
                sentences[indx] = list(e.keys())[0]
            else:
                sentences[indx] = ''
            sentences[indx+1] = node
            sentences[indx+2] = data['node'][node]['text']
            chosen_node_ids.append(node)
            scores.append(score)
            indx += 3
            retries=0
        else:
            retries+=1
        if retries > walk_len:
            break
 
        node_id = node

    return sentences, label, chosen_node_ids, extended_edges,sum(scores)/len(scores) if scores else 0







def calculate_longest_path(extended_edges, chosen_node_ids, label, starting_node_list):
    """Calculates the depth (longest path length) of an undirected subgraph 
    defined by the extended_edges and chosen_node_ids."""
    
    # Initialize an undirected graph
    subgraph = nx.Graph()

    if label != 'Graph Depth':
        subgraph = nx.DiGraph() # random walk is directed
    
    # Add edges to the subgraph as undirected edges and keep original order
    for child, parent in extended_edges:
        if child in chosen_node_ids and parent in chosen_node_ids:
            subgraph.add_edge(child, parent)
    
    # Function to perform DFS and find the longest path from a given node
    def dfs(node, visited):
        visited.add(node)
        max_depth = 0
        for neighbor in subgraph.neighbors(node):
            if neighbor not in visited:
                depth = 1 + dfs(neighbor, visited)
                max_depth = max(max_depth, depth)
        visited.remove(node)# Unmark node to allow other paths to reuse it in other DFS calls
        return max_depth
    
    # Try DFS from each node to find the longest path in the subgraph
    longest_path_length = 0
    for start_node in starting_node_list: #chosen_node_ids:
        if not subgraph.has_node(start_node):
            continue
        longest_path_length = max(longest_path_length, dfs(start_node, set()))
    
    print('longest path for a '+str(label)+'='+str(longest_path_length))

    return longest_path_length



# ==============================
# Paths
# ==============================
GUEST_PATH = 'Guest/random-walks/eacl_graphs/'
OUTPUT_DIR = 'Guest/compare2/random-walks/stats'

model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

# ==============================
# Graph statistics function
# ==============================
def compute_graph_stats(data):
    """
    Compute structural statistics for a Guest discourse graph.
    Assumes each node has at most ONE parent.
    """
    G = nx.DiGraph()
    for node_id in data['node'].keys():
        G.add_node(node_id)
    for child_id in data['edge'].keys():
        parent_edges = data['edge'][child_id]
        if len(parent_edges) > 0:
            parent_id = list(parent_edges.keys())[0]
            if parent_id in data['node']:
                G.add_edge(parent_id, child_id)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    try:
        depth = nx.dag_longest_path_length(G)
    except Exception:
        depth = 0

    out_degrees = [G.out_degree(n) for n in G.nodes() if G.out_degree(n) > 0]
    max_breadth = max(out_degrees) if out_degrees else 0
    mean_breadth = np.mean(out_degrees) if out_degrees else 0

    return {
        "nodes": num_nodes,
        "edges": num_edges,
        "depth": depth,
        "max_breadth": max_breadth,
        "mean_breadth": mean_breadth
    }

# ==============================
# Initialize collections
# ==============================
all_stats = []
longest_paths_all = []
mean_branching_factors_all = []

dataset_samples = []
dataset_samples1 = []

files = os.listdir(GUEST_PATH)
print(f"Processing {len(files)} Guest graphs...")

# ==============================
# Loop over graphs
# ==============================
for indx, file in enumerate(files):
    print("processing",indx)

    with open(os.path.join(GUEST_PATH, file), "rb") as f:
        data = pkl.load(f)

    # Compute and store per-graph statistics
    stats = compute_graph_stats(data)
    stats["graph_id"] = file
    all_stats.append(stats)

    # Build child edges for random walks
    child_edges = {}
    edges = []
    for node_id in data['node'].keys():
        edge = data['edge'][node_id]
        if len(edge.keys()) > 0:
            parent = list(edge.keys())[0]
            if parent in child_edges:
                child_edges[parent].append(node_id)
            else:
                child_edges[parent] = [node_id]
            edges.append([parent, node_id])

    # Walks per node
    for node_id in data['node'].keys():

        # 1D walk first
        sentences1 = ['']*walk_length*3
        sentences1, label1, chosen_node_ids1, extended_edges1, score1 = one_d_random_graph_walk(
            sentences1, data, node_id, child_edges, walk_length)

        # 2D walk
        sentences = ['']*walk_length*3
        sentences, label, chosen_node_ids, extended_edges, score = random_graph_walk(
            sentences, data, node_id, child_edges, walk_length)

        if label != -1 and label != '' and not sentences[5] == '' and not sentences1[5] == '':
            # Save samples
            sentences.append(label)
            sentences1.append(label1)
            dataset_samples.append(sentences)
            dataset_samples1.append(sentences1)

            # Depth metric
            longest_path_length1 = calculate_longest_path(extended_edges1, chosen_node_ids1,label,[chosen_node_ids1[0]])
            longest_path_length = calculate_longest_path(extended_edges, chosen_node_ids,label,[chosen_node_ids[0]])
            longest_paths_all.append((longest_path_length1, longest_path_length))

            # Mean branching metric
            out_degree_count1 = defaultdict(int)
            for parent, child in extended_edges1:
                if parent in chosen_node_ids1 and child in chosen_node_ids1:
                    out_degree_count1[parent] += 1
            mean_branching_factor1 = sum(out_degree_count1.values())/len(out_degree_count1) if out_degree_count1 else 0

            out_degree_count = defaultdict(int)
            for parent, child in extended_edges:
                if parent in chosen_node_ids and child in chosen_node_ids:
                    out_degree_count[parent] += 1
            mean_branching_factor = sum(out_degree_count.values())/len(out_degree_count) if out_degree_count else 0

            mean_branching_factors_all.append((mean_branching_factor1, mean_branching_factor))

# ==============================
# Save per-graph statistics CSV
# ==============================
df = pd.DataFrame(all_stats)
df.to_csv(os.path.join(OUTPUT_DIR, "guest_graph_statistics_raw.csv"), index=False)

summary = pd.DataFrame({
    "mean_nodes": [df["nodes"].mean()],
    "std_nodes": [df["nodes"].std()],
    "mean_edges": [df["edges"].mean()],
    "std_edges": [df["edges"].std()],
    "mean_depth": [df["depth"].mean()],
    "std_depth": [df["depth"].std()],
    "max_depth": [df["depth"].max()],
    "mean_max_breadth": [df["max_breadth"].mean()],
    "std_max_breadth": [df["max_breadth"].std()],
    "max_breadth": [df["max_breadth"].max()],
    "mean_breadth": [df["mean_breadth"].mean()],
    "std_mean_breadth": [df["mean_breadth"].std()],
    "max_mean_breadth": [df["mean_breadth"].max()],

})
summary.to_csv(os.path.join(OUTPUT_DIR, "guest_graph_statistics_summary.csv"), index=False)
print("Saved graph statistics CSVs.")

# ==============================
# Plot 1D and 2D histograms (Nature-style 2x2)
# ==============================

import seaborn as sns

sns.set_theme(style="whitegrid")  # Nature-style background
colors = ['#4C9A8A', '#C44E52']  # muted colors

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
panel_letters = ['A', 'B', 'C', 'D']

# ==============================
# Prepare integer-centered bins
# ==============================

depth_1d = [x[0] for x in longest_paths_all]
depth_2d = [x[1] for x in longest_paths_all]

breadth_1d = [x[0] for x in mean_branching_factors_all]
breadth_2d = [x[1] for x in mean_branching_factors_all]


def shared_integer_bins(values_a, values_b):
    vmin = int(min(np.min(values_a), np.min(values_b)))
    vmax = int(max(np.max(values_a), np.max(values_b)))

    # Integer-centered bins
    bins = np.arange(vmin - 0.5, vmax + 1.5, 1)
    ticks = np.arange(vmin, vmax + 1, 1)

    return bins, ticks, vmin, vmax


# Shared bins for depth
bins_depth, ticks_depth, xmin_d, xmax_d = shared_integer_bins(
    depth_1d, depth_2d
)

# Shared bins for breadth
bins_breadth, ticks_breadth, xmin_b, xmax_b = shared_integer_bins(
    breadth_1d, breadth_2d
)


# ==============================
# Depth 1D
# ==============================
counts, bins, patches = axes[0, 0].hist(
    depth_1d,
    bins=bins_depth,
    alpha=0.7,
    color=colors[0],
    edgecolor='black'
)

for count, patch in zip(counts, patches):
    if count > 0:
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()
        axes[0, 0].text(
            x, y,
            f'{int(count)}',
            ha='center',
            va='bottom',
            fontsize=16,
        )

axes[0, 0].set_title(panel_letters[0], x=-0.08, loc='left',
                     fontweight='bold', fontsize=16)
axes[0, 0].set_xlabel('1D Depth', fontsize=18, fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontsize=18, fontweight='bold')
axes[0, 0].set_xticks(ticks_depth)
axes[0, 0].set_xlim(xmin_d - 0.5, xmax_d + 0.5)
axes[0, 0].tick_params(axis='both', labelsize=16)


# ==============================
# Depth 2D
# ==============================
counts, bins, patches = axes[0, 1].hist(
    depth_2d,
    bins=bins_depth,
    alpha=0.7,
    color=colors[1],
    edgecolor='black'
)

for count, patch in zip(counts, patches):
    if count > 0:
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()
        axes[0, 1].text(
            x, y,
            f'{int(count)}',
            ha='center',
            va='bottom',
            fontsize=16,
        )

axes[0, 1].set_title(panel_letters[1], x=-0.08, loc='left',
                     fontweight='bold', fontsize=16)
axes[0, 1].set_xlabel('2D Depth', fontsize=18, fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontsize=18, fontweight='bold')
axes[0, 1].set_xticks(ticks_depth)
axes[0, 1].set_xlim(xmin_d - 0.5, xmax_d + 0.5)
axes[0, 1].tick_params(axis='both', labelsize=16)


# ==============================
# Breadth 1D
# ==============================
counts, bins, patches = axes[1, 0].hist(
    breadth_1d,
    bins=bins_breadth,
    alpha=0.7,
    color=colors[0],
    edgecolor='black'
)

for count, patch in zip(counts, patches):
    if count > 0:
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()
        axes[1, 0].text(
            x, y,
            f'{int(count)}',
            ha='center',
            va='bottom',
            fontsize=16,
        )

axes[1, 0].set_title(panel_letters[2], x=-0.08, loc='left',
                     fontweight='bold', fontsize=16)
axes[1, 0].set_xlabel('1D Breadth', fontsize=18, fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontsize=18, fontweight='bold')
axes[1, 0].set_xticks(ticks_breadth)
axes[1, 0].set_xlim(xmin_b - 0.5, xmax_b + 0.5)
axes[1, 0].tick_params(axis='both', labelsize=16)


# ==============================
# Breadth 2D
# ==============================
counts, bins, patches = axes[1, 1].hist(
    breadth_2d,
    bins=bins_breadth,
    alpha=0.7,
    color=colors[1],
    edgecolor='black'
)

for count, patch in zip(counts, patches):
    if count > 0:
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()
        axes[1, 1].text(
            x, y,
            f'{int(count)}',
            ha='center',
            va='bottom',
            fontsize=16,
        )

axes[1, 1].set_title(panel_letters[3], x=-0.08, loc='left',
                     fontweight='bold', fontsize=16)
axes[1, 1].set_xlabel('2D Breadth', fontsize=18, fontweight='bold')
axes[1, 1].set_ylabel('Frequency', fontsize=18, fontweight='bold')
axes[1, 1].set_xticks(ticks_breadth)
axes[1, 1].set_xlim(xmin_b - 0.5, xmax_b + 0.5)
axes[1, 1].tick_params(axis='both', labelsize=16)



# ==============================
# Style adjustments (Nature-like)
# ==============================
for ax in axes.flatten():
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.25)



plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'depth_branching_histograms.png'), dpi=300)
plt.savefig(os.path.join(OUTPUT_DIR, 'depth_branching_histograms.pdf'), dpi=1000, format="pdf",bbox_inches='tight')

plt.show()

# ==============================
# Paired t-tests
# ==============================
depth_1d = [x[0] for x in longest_paths_all]
depth_2d = [x[1] for x in longest_paths_all]
branch_1d = [x[0] for x in mean_branching_factors_all]
branch_2d = [x[1] for x in mean_branching_factors_all]

t_depth, p_depth = ttest_rel(depth_1d, depth_2d)
t_branch, p_branch = ttest_rel(branch_1d, branch_2d)

print(f"Paired t-test Depth 1D vs 2D: t={t_depth:.3f}, p={p_depth:.3e}")
print(f"Paired t-test Branching 1D vs 2D: t={t_branch:.3f}, p={p_branch:.3e}")

# File path to save the t-test results
ttest_file = "Guest/compare2/random-walks/stats/paired_ttest_results.txt"

# Open the file in write mode and write the results
with open(ttest_file, "w") as f:
    f.write(f"Paired t-test Depth 1D vs 2D: t={t_depth:.3f}, p={p_depth:.3e}\n")
    f.write(f"Paired t-test Branching 1D vs 2D: t={t_branch:.3f}, p={p_branch:.3e}\n")

print(f"T-test results saved to {ttest_file}")



if p_depth < 0.05:
    print("Depth difference is statistically significant (p<0.05)")
else:
    print("Depth difference is not statistically significant (p>=0.05)")

if p_branch < 0.05:
    print("Branching difference is statistically significant (p<0.05)")
else:
    print("Branching difference is not statistically significant (p>=0.05)")
