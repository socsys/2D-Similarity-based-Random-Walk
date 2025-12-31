"""Similarity-based Random Walk."""

import os
import csv
import pickle as pkl
import pandas as pd
import numpy as np
import random
import math
from sentence_transformers import SentenceTransformer, util
from statsmodels.nonparametric.smoothers_lowess import lowess
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



def zero_d_random_graph_walk(sentences, data, node_id, child_edges, walk_len):

    #print('generating new 1D graph walk')

    #print('Main Comment:', [node_id,data.node[node_id]['text']] )
    #print('Category:',data.node[node_id]['label'])

    #some cleaning first to replace all nans
    for n_id in data.node:
        for key, value in data.node[n_id].items():
            if key=='text' and pd.isna(value):  # Check for NaN
                data.node[n_id][key] = ''  # Replace NaN with empty string

    edge= data.edge[node_id]
    if len(edge.keys()) > 0:
        sentences[0] = list(edge.keys())[0] #parent id
    else:
        sentences[0]=''
        return sentences, -1, [], [],0 # we must have a parent to get the label

    sentences[1] = node_id
    sentences[2] = data.node[node_id]['text']    
    weight=edge[list(edge.keys())[0]]['weight']
    label = weight if weight==1 else 0
    chosen_node_ids = [node_id]
    indx = 3
    retries = 0
    original_node_id=node_id
    scores=[] #relevance scores

    extended_edges=[] # we do not extend same edge twice

    # Adding parent mandatorily
    edge = data.edge[node_id]
    if len(edge.keys()) > 0:
        child_id=node_id
        node_id = list(edge.keys())[0] # there is only one parent
        if node_id not in data.node and node_id not in data.edge:
            return sentences, label, chosen_node_ids, extended_edges,0 #relevance score is 0
        else:
            e=data.edge[node_id]
            if len(e.keys()) > 0:
                sentences[3] = list(e.keys())[0]
            else:
                sentences[3] = ''
            sentences[4] = node_id
            sentences[5] = data.node[node_id]['text']
            chosen_node_ids.append(node_id)
            extended_edges.append([child_id, node_id]) # save to extended edges
            indx += 3
    ########

        
    return sentences, label, chosen_node_ids, extended_edges,sum(scores)/len(scores) if scores else 0



def one_d_random_graph_walk(sentences, data, node_id, child_edges, walk_len):

    #print('generating new 1D graph walk')

    #print('Main Comment:', [node_id,data.node[node_id]['text']] )
    #print('Category:',data.node[node_id]['label'])

    #some cleaning first to replace all nans
    for n_id in data.node:
        for key, value in data.node[n_id].items():
            if key=='text' and pd.isna(value):  # Check for NaN
                data.node[n_id][key] = ''  # Replace NaN with empty string

    edge= data.edge[node_id]
    if len(edge.keys()) > 0:
        sentences[0] = list(edge.keys())[0] #parent id
    else:
        sentences[0]=''
        return sentences, -1, [], [],0 # we must have a parent to get the label

    sentences[1] = node_id
    sentences[2] = data.node[node_id]['text']    
    weight=edge[list(edge.keys())[0]]['weight']
    label = weight if weight==1 else 0
    chosen_node_ids = [node_id]
    indx = 3
    retries = 0
    original_node_id=node_id
    scores=[] #relevance scores

    extended_edges=[] # we do not extend same edge twice

    # Adding parent mandatorily
    edge = data.edge[node_id]
    if len(edge.keys()) > 0:
        child_id=node_id
        node_id = list(edge.keys())[0] # there is only one parent
        if node_id not in data.node and node_id not in data.edge:
            return sentences, label, chosen_node_ids, extended_edges,0 #relevance score is 0
        else:
            e=data.edge[node_id]
            if len(e.keys()) > 0:
                sentences[3] = list(e.keys())[0]
            else:
                sentences[3] = ''
            sentences[4] = node_id
            sentences[5] = data.node[node_id]['text']
            chosen_node_ids.append(node_id)
            extended_edges.append([child_id, node_id]) # save to extended edges
            indx += 3
    ########

    while indx < walk_len*3: 

        choices = []
        choices_text = []

        #visited_nodes.append(node_id)
        edge = data.edge[node_id]

        #parent node
        if len(edge.keys()) > 0 and list(edge.keys())[0] in data.node.keys(): 
            parent_id=list(edge.keys())[0]
            choices.append(parent_id) #append parent
            choices_text.append([parent_id,data.node[parent_id]['text']])
            if [node_id,parent_id] not in extended_edges and [parent_id,node_id] not in extended_edges:
                extended_edges.append([node_id,parent_id]) # save to extended edges

        if node_id in child_edges:
            for child_id in child_edges[node_id]: #there are possibly multiple children
                if child_id in data.node.keys(): 
                    choices.append(child_id) #append child
                    choices_text.append([child_id,data.node[child_id]['text']])
                    if [node_id,child_id] not in extended_edges and [child_id,node_id] not in extended_edges:
                        extended_edges.append([node_id,child_id]) # save to extended edges
      
        #print('length of choices=',len(choices))

        if len(choices) == 0:
            return sentences, label, chosen_node_ids, extended_edges,sum(scores)/len(scores) if scores else 0

        
        hits = sbert_cosine_similarity(
            data.node[node_id]['text'], [t[1] for t in choices_text], len(choices))
        
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
            data.node[original_node_id]['text'], [choices_text[selected_corpus_id][1]], 1)
        score=original_node_relevance[0]['score']

        # print('selected node:',node)
        # print('score',score)

        # print('************** 1D *****************************')


        if node not in chosen_node_ids:
            e=data.edge[node]
            if len(e.keys()) > 0:
                sentences[indx] = list(e.keys())[0]
            else:
                sentences[indx] = ''
            sentences[indx+1] = node
            sentences[indx+2] = data.node[node]['text']
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



def random_graph_walk(sentences, data, node_id, child_edges, walk_len):

    #print('generating new 2D graph walk')

    #print('Main Comment:', [node_id,data.node[node_id]['text']] )
    #print('Category:',data.node[node_id]['label'])

    #some cleaning first to replace all nans
    for n_id in data.node:
        for key, value in data.node[n_id].items():
            if key=='text' and pd.isna(value):  # Check for NaN
                data.node[n_id][key] = ''  # Replace NaN with empty string

    retries=0
    edge= data.edge[node_id]
    if len(edge.keys()) > 0:
        sentences[0] = list(edge.keys())[0] #parent id
    else:
        sentences[0]=''
        return sentences, -1, [], [],0 # we must have a parent to get the label
    sentences[1] = node_id # node id
    sentences[2] = data.node[node_id]['text']#node text
    weight=edge[list(edge.keys())[0]]['weight']
    label = weight if weight==1 else 0
    votes=data.node[node_id]['votes']
    # print(votes)
    # print(compute_votes_score(votes))
    #sys.exit() 
    chosen_node_ids = [node_id]
    visited_nodes=[node_id]
    indx = 3
    original_node_id=node_id
    scores=[] #relevance scores

    extended_edges=[] # we do not extend same edge twice

    # Adding parent mandatorily
    edge = data.edge[node_id]
    if len(edge.keys()) > 0:
        child_id=node_id
        node_id = list(edge.keys())[0] # there is only one parent
        if node_id not in data.node and node_id not in data.edge:
            return sentences, label, chosen_node_ids, extended_edges,0 #relevance score is 0
        else:
            e=data.edge[node_id]
            if len(e.keys()) > 0:
                sentences[3] = list(e.keys())[0]
            else:
                sentences[3] = ''
            sentences[4] = node_id
            sentences[5] = data.node[node_id]['text'] 
            chosen_node_ids.append(node_id)
            extended_edges.append([child_id, node_id]) # save to extended edges
            indx += 3
    ########
    choices = []
    choices_text = []
 
    while indx < walk_len*3: 

        visited_nodes.append(node_id)
        edge = data.edge[node_id]

        #parent node
        # and not list(edge.keys())[0] in visited_nodes
        if len(edge.keys()) > 0 and list(edge.keys())[0] in data.node.keys(): 
            parent_id=list(edge.keys())[0]
            if not parent_id in choices:
                choices.append(parent_id) #append parent
                choices_text.append([parent_id,data.node[parent_id]['text']])
            if not [node_id,parent_id] in extended_edges and not [parent_id,node_id] in extended_edges:
                extended_edges.append([node_id,parent_id]) # save to extended edges

        if node_id in child_edges:
            for child_id in child_edges[node_id]: #there are possibly multiple children
                #not child_id in visited_nodes and
                if  child_id in data.node.keys(): 
                    if not child_id in choices:
                        choices.append(child_id) #append child
                        choices_text.append([child_id,data.node[child_id]['text']])
                        if not [node_id,child_id] in extended_edges and not [child_id,node_id] in extended_edges:
                            extended_edges.append([node_id,child_id]) # save to extended edges
      


        if len(choices) == 0:
            return sentences, label, chosen_node_ids, extended_edges, sum(scores)/len(scores) if scores else 0
        
        hits = sbert_cosine_similarity(
            data.node[original_node_id]['text'], [t[1] for t in choices_text], len(choices))
        
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


        if node not in chosen_node_ids and not data.node[node]['text']=='' :
            e=data.edge[node]
            if len(e.keys()) > 0:
                sentences[indx] = list(e.keys())[0]
            else:
                sentences[indx] = ''
            sentences[indx+1]= node #child
            sentences[indx+2] = data.node[node]['text'] #text
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
        choices_text.remove([node_id,data.node[node_id]['text']])
        #print('length of choices=',len(choices))
        #debugging
        # print('**********debugging***********************')
        #print('selected choice', [node_id,data.node[node_id]['text']] )
        #print('indx=',indx,', choices=',choices)
        # print('choices_text=',choices_text) 
        # print('******************************************')

        
    return sentences, label, chosen_node_ids, extended_edges,sum(scores)/len(scores) if scores else 0


def random_L_utterances(sentences, data, node_id, L):
    """
    Baseline (i): Random L utterances from the graph (new 1D/2D format).
    Includes mandatory parent of the first node if available.
    """
    #print("Generating random L utterances (with mandatory parent)")

    # Clean NaNs
    for n_id in data.node:
        for key, value in data.node[n_id].items():
            if key == 'text' and pd.isna(value):
                data.node[n_id][key] = ''

    label = -1
    extended_edges = []
    chosen_node_ids = []
    idx = 0
    to_be_added = L

    # 1) Add original node + mandatory parent
    edge = data.edge.get(node_id, {})
    if len(edge) > 0:
        parent_id = list(edge.keys())[0]
        parent_text = data.node[parent_id]['text']
        sentences[idx] = parent_id
    else:
        parent_id = ""
        parent_text = ""
        sentences[idx] = ""
        return sentences, label, chosen_node_ids, extended_edges, 0

    sentences[idx+1] = node_id
    sentences[idx+2] = data.node[node_id]['text']
    chosen_node_ids.append(node_id)
    weight = edge[list(edge.keys())[0]]['weight']
    label = weight if weight == 1 else 0
    idx += 3
    to_be_added -= 1

    # Add parent node if exists
    if parent_id != "":
        parent_of_parent = list(data.edge[parent_id].keys())[0] if len(data.edge[parent_id]) > 0 else ""
        sentences[idx] = parent_of_parent
        sentences[idx+1] = parent_id
        sentences[idx+2] = parent_text
        chosen_node_ids.append(parent_id)
        extended_edges.append([node_id, parent_id])
        idx += 3
        to_be_added -= 1

    # 2) Randomly pick nodes excluding original + parent
    all_nodes = [n for n in data.node if n not in [node_id, parent_id]]
    #print("Kialo: all node size=",len(all_nodes))
    picked = random.sample(all_nodes, min(to_be_added, len(all_nodes)))

    for nid in picked:
        parent = list(data.edge[nid].keys())[0] if len(data.edge[nid]) > 0 else ""
        text = data.node[nid]['text']
        sentences[idx] = parent
        sentences[idx+1] = nid
        sentences[idx+2] = text
        chosen_node_ids.append(nid)
        idx += 3
        to_be_added -= 1

    #print(idx/3,"nodes were added as random L")

    return sentences, label, chosen_node_ids, extended_edges, 0


def top_L_relevant_utterances(sentences, data, node_id, L):
    """
    Baseline (ii): Top L most relevant utterances (new 1D/2D format).
    Includes mandatory parent of the first node if available.
    """
    #print("Generating top L relevant utterances (with mandatory parent)")

    # Clean NaNs
    for n_id in data.node:
        for key, value in data.node[n_id].items():
            if key == 'text' and pd.isna(value):
                data.node[n_id][key] = ''

    label = -1
    extended_edges = []
    chosen_node_ids = []
    idx = 0
    to_be_added = L
    original_text = data.node[node_id]['text']

    # 1) Add original node + mandatory parent
    edge = data.edge.get(node_id, {})
    if len(edge) > 0:
        parent_id = list(edge.keys())[0]
        parent_text = data.node[parent_id]['text']
        sentences[idx] = parent_id
    else:
        parent_id = ""
        parent_text = ""
        sentences[idx] = ""
        return sentences, label, chosen_node_ids, extended_edges, 0

    sentences[idx+1] = node_id
    sentences[idx+2] = original_text
    weight = edge[list(edge.keys())[0]]['weight']
    label = weight if weight == 1 else 0
    chosen_node_ids.append(node_id)
    idx += 3
    to_be_added -= 1

    # Add parent node if exists
    if parent_id != "":
        parent_of_parent = list(data.edge[parent_id].keys())[0] if len(data.edge[parent_id]) > 0 else ""
        sentences[idx] = parent_of_parent
        sentences[idx+1] = parent_id
        sentences[idx+2] = parent_text
        chosen_node_ids.append(parent_id)
        extended_edges.append([node_id, parent_id])
        idx += 3
        to_be_added -= 1

    # 2) Prepare candidate nodes excluding original + parent
    candidates = []
    for nid in data.node:
        if nid in [node_id, parent_id]:
            continue
        text = data.node[nid]['text']
        if text != "":
            candidates.append((nid, text))

    if not candidates:
        return sentences, label, chosen_node_ids, extended_edges, 0

    ids = [c[0] for c in candidates]
    texts = [c[1] for c in candidates]

    hits = sbert_cosine_similarity(original_text, texts, len(texts))

    # Attach id and text
    for i, h in enumerate(hits):
        corpus_index = h['corpus_id']
        h['node_id'] = ids[corpus_index]
        h['text'] = texts[corpus_index]

    # Pretty print
    # for h in hits:
    #     print("target node text:", sentences[2])
    #     print(f"ID: {h['node_id']}\nScore: {h['score']}\nText: {h['text']}\n---")

    # Sort and take top L
    hits_sorted = sorted(hits, key=lambda x: x['score'], reverse=True)
    top_hits = hits_sorted[:to_be_added]
    scores = []

    for h in top_hits:
        nid = h['node_id']
        score = h['score']
        text = data.node[nid]['text']
        parent = list(data.edge[nid].keys())[0] if len(data.edge[nid]) > 0 else ""

        sentences[idx] = parent
        sentences[idx+1] = nid
        sentences[idx+2] = text
        chosen_node_ids.append(nid)
        scores.append(score)
        idx += 3
        to_be_added -= 1
        #print("Top-L to-be-added remaining:", to_be_added)

    avg_score = sum(scores) / len(scores) if scores else 0

    return sentences, label, chosen_node_ids, extended_edges, avg_score


# =========================
# Configuration
# =========================
model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)
dataset_path = 'Kialo/Classification/random-walks/serializedGraphs/' #  or provide the dataset path C:\Users\za0005\OneDrive - University of Surrey\Desktop\Surrey\NLI\Datasets\kialo_debates\serializedGraphs
files = os.listdir(dataset_path)

MAX_FILES = 80
WALK_LENGTH = walk_length  # assumes already defined
DATASET_PATH = dataset_path
FILES = files  # assumes already defined

# =========================
# Runtime containers
# =========================
random_walk_runtime = defaultdict(list)
zeroD_walk_runtime = defaultdict(list)
oneD_walk_runtime = defaultdict(list)
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

    num_nodes = len(data.node)

    # Just for Random Walk.
    child_edges = {}
    edges=[]
    for node_id in data.node.keys():
        edge = data.edge[node_id]
        if len(edge.keys()) > 0:
            parent = list(edge.keys())[0]
            if parent in child_edges:
                child_edges[parent].append(node_id)
            else:
                child_edges[parent] = [node_id]
            edges.append([parent,node_id]) # simple data structure for traversal (e.g., for computing graph depth)
            

    for node_id in data.node.keys():

        if len(twoD_walk_runtime[num_nodes])>30:
            break


        print('Processing', indx, file)


        start_random = time.time()
        sentences = [''] * WALK_LENGTH * 3
        random_L_utterances(
            sentences,
            data,
            node_id,
            WALK_LENGTH
        )
        end_random = time.time()


        start_0d = time.time()
        sentences = [''] * WALK_LENGTH * 3
        zero_d_random_graph_walk(
            sentences,
            data,
            node_id,
            child_edges,
            WALK_LENGTH
        )
        end_0d = time.time()

        start_1d = time.time()
        sentences = [''] * WALK_LENGTH * 3
        one_d_random_graph_walk(
            sentences,
            data,
            node_id,
            child_edges,
            WALK_LENGTH
        )
        end_1d = time.time()


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

        random_walk_runtime[num_nodes].append(end_random-start_random)
        zeroD_walk_runtime[num_nodes].append(end_0d-start_0d)
        oneD_walk_runtime[num_nodes].append(end_1d-start_1d)
        twoD_walk_runtime[num_nodes].append(end_2d - start_2d)
        top_walk_runtime[num_nodes].append(end_top - start_top)




# =========================
# Aggregate mean runtime
# =========================
graph_sizes = sorted(twoD_walk_runtime.keys())

random_means=[np.mean(random_walk_runtime[n]) for n in graph_sizes]
zeroD_means=[np.mean(zeroD_walk_runtime[n]) for n in graph_sizes]
oneD_means=[np.mean(oneD_walk_runtime[n]) for n in graph_sizes]
twoD_means = [np.mean(twoD_walk_runtime[n]) for n in graph_sizes]
top_means  = [np.mean(top_walk_runtime[n]) for n in graph_sizes]


# =========================
# Plot (Nature-style)
# =========================
plt.figure(figsize=(7, 5))

# LOWESS smoothing (visual only)
# twoD_smooth = lowess(twoD_means, graph_sizes, frac=0.3, return_sorted=False)
# top_smooth  = lowess(top_means, graph_sizes, frac=0.3, return_sorted=False)

# Optional: show raw points faintly
# plt.scatter(graph_sizes, twoD_means, color='#2C7FB8', alpha=0.25)
# plt.scatter(graph_sizes, top_means, color='#B2182B', alpha=0.25)

# Smoothed curves
plt.plot(
    graph_sizes,
    random_means,
    linewidth=3,
    color='#2ca3f5',  # Blue for Random baseline
    label='Random Walk'
)
plt.plot(
    graph_sizes,
    zeroD_means,
    linewidth=3,
    linestyle="--",
    color="#000000",  # for 0D baseline
    label='0D Walk'
)
plt.plot(
    graph_sizes,
    oneD_means,
    linewidth=3,
    color='#4bdbbe',  # 1D Walk (teal)
    label='1D Walk'
)
plt.plot(
    graph_sizes,
    twoD_means,
    linewidth=3,
    color='#F27276',  # 2D Walk (red)
    label='2D Walk'
)
plt.plot(
    graph_sizes,
    top_means,
    linestyle='-.',
    linewidth=3,
    color="#800000",  # Top-L Baseline (dark red)
    label='Top-L Baseline'
)


plt.xlabel('Number of Nodes', fontsize=18, fontweight='bold')
plt.ylabel('Runtime (Seconds)', fontsize=18, fontweight='bold')


ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))


plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.legend(fontsize=16, frameon=False)

plt.tight_layout()


plt.savefig(f'Kialo/Classification/compare2/random-walks/runtime/Kialo_runtime.png', dpi=300)
plt.savefig(f'Kialo/Classification/compare2/random-walks/runtime/Kialo_runtime.pdf', format='pdf', dpi=1000, bbox_inches="tight")

plt.show()
