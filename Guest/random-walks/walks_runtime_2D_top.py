"""Similarity-based Random Walk."""

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
import time
import matplotlib.lines as mlines
from statistics import mean
from matplotlib.ticker import MaxNLocator


walk_length=10
M=walk_length #maximum retries to revisit a node before terminating any walk
population=400
plot_dist_after=100
plot_early=False
plot_end=False
save_data=True

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
            if retries>M:
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

def top_L_relevant_utterances(sentences, data, node_id, L):
    """
    Baseline (ii): Top L most relevant utterances to the original node.
    Now includes mandatory parent of the first node if available.
    """

    print('generating baseline: top L relevant utterances (with mandatory parent)')

    # Clean NaNs
    for n_id in data['node']:
        if pd.isna(data['node'][n_id]['text']):
            data['node'][n_id]['text'] = ''

    label = data['node'][node_id]['label']
    original_text = data['node'][node_id]['text']

    ##########################################
    # 1) Insert original node
    ##########################################
    parent_edge = data['edge'][node_id]
    if len(parent_edge.keys()) > 0:
        parent_id = list(parent_edge.keys())[0]
        sentences[0] = parent_id
    else:
        parent_id = ""
        sentences[0] = ""

    sentences[1] = node_id
    sentences[2] = original_text

    chosen_node_ids = [node_id]
    extended_edges = []
    idx = 3


    ##########################################
    # 2) Compute relevance ranking for all other nodes
    ##########################################
    candidates = []
    print("all nodes size:", len(data['node'])-1)
    for nid in data['node']:
        if nid == node_id:
            continue
        text = data['node'][nid]['text']
        if text != "":
            candidates.append((nid, text))

    if not candidates:
        return sentences, label, chosen_node_ids, extended_edges, 0

    ids = [c[0] for c in candidates]
    texts = [c[1] for c in candidates]

    hits = sbert_cosine_similarity(original_text, texts, len(texts))

    # Attach id AND text
    for i, h in enumerate(hits):
        corpus_index = h['corpus_id']
        h['node_id'] = ids[corpus_index]
        h['text'] = texts[corpus_index]     
        

    # Sort by score
    hits_sorted = sorted(hits, key=lambda x: x['score'], reverse=True)

    # Take top L
    top_hits = hits_sorted[:L-1]

    scores = []

    for h in top_hits:
        nid = h['node_id']
        score = h['score']
        text = data['node'][nid]['text']
        parent = list(data['edge'][nid].keys())[0] if len(data['edge'][nid]) > 0 else ""

        sentences[idx] = parent
        sentences[idx+1] = nid
        sentences[idx+2] = text

        chosen_node_ids.append(nid)
        scores.append(score)
        idx += 3

    print("added:",idx/3, "nodes" )

    avg_score = sum(scores)/len(scores) if scores else 0

    return sentences, label, chosen_node_ids, extended_edges, avg_score



# =========================
# Configuration
# =========================
model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)
dataset_path = 'Guest/random-walks/eacl_graphs/' #  or provide the dataset path C:\Users\za0005\OneDrive - University of Surrey\Desktop\Surrey\NLI\Datasets\kialo_debates\serializedGraphs
files = os.listdir(dataset_path)

MAX_FILES = 200
WALK_LENGTH = walk_length  # assumes already defined
DATASET_PATH = dataset_path
FILES = files  # assumes already defined

# =========================
# Runtime containers
# =========================
twoD_walk_runtime = defaultdict(list)
top_walk_runtime = defaultdict(list)

# =========================
# Runtime measurement
# =========================
for indx, file in enumerate(FILES):
    if indx == MAX_FILES:
        break

    print(f'Processing file {indx}')
    data = pkl.load(open(DATASET_PATH + file, 'rb'))

    num_nodes = len(data['node'])

    # Build child edges once per graph
    child_edges = {}
    edges=[]
    for node_id in data['node'].keys():
        edge = data['edge'][node_id]
        if len(edge.keys()) > 0:
            parent = list(edge.keys())[0]
            if parent in child_edges:
                child_edges[parent].append(node_id)
            else:
                child_edges[parent] = [node_id]
            edges.append([parent,node_id]) # simple data structure for traversal (e.g., for computing graph depth)
            
    # Measure runtime per node
    for node_id in data['node'].keys():

        # ---------- 2D Walk ----------
        start_2d = time.time()
        sentences = [''] * WALK_LENGTH * 3
        random_graph_walk(
            sentences,
            data,
            node_id,
            child_edges,
            WALK_LENGTH
        )
        end_2d = time.time()

        # ---------- Top-L Baseline ----------
        start_top = time.time()
        sentences1 = [''] * WALK_LENGTH * 3
        top_L_relevant_utterances(
            sentences1,
            data,
            node_id,
            WALK_LENGTH
        )
        end_top = time.time()

        twoD_walk_runtime[num_nodes].append(end_2d - start_2d)
        top_walk_runtime[num_nodes].append(end_top - start_top)

# =========================
# Aggregate mean runtime
# =========================
graph_sizes = sorted(twoD_walk_runtime.keys())

twoD_means = [np.mean(twoD_walk_runtime[n]) for n in graph_sizes]
top_means  = [np.mean(top_walk_runtime[n]) for n in graph_sizes]

# =========================
# Plot (Nature-style)
# =========================
plt.figure(figsize=(7, 5))

# Color-blind safe colors (Okabe–Ito palette)
plt.plot(
    graph_sizes,
    twoD_means,
    marker='o',
    linewidth=2.5,
    color='#0072B2',
    label='2D Walk'
)

plt.plot(
    graph_sizes,
    top_means,
    marker='s',
    linestyle='--',
    linewidth=2.5,
    color='#D55E00',
    label='Top-L Baseline'
)

plt.xlabel('Number of nodes', fontsize=18, fontweight='bold')
plt.ylabel('Runtime (seconds)', fontsize=18, fontweight='bold')


ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))


plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.legend(fontsize=16, frameon=False)

plt.tight_layout()


plt.savefig(f'Guest/compare2/random-walks/runtime/Guest_2D_vs_TopL_runtime.png', dpi=300)
plt.savefig(f'Guest/compare2/random-walks/runtime/Guest_2D_vs_TopL_runtime.pdf', format='pdf', dpi=1000, bbox_inches="tight")

plt.show()
