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


walk_length=10
population=50
plot_dist_after=30000
plot_early=True
plot_end=False
save_data=False

label_map={0:'Attacking',1:'Supporting'}

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


def compute_votes_score(votes):
    return sum([index*v for index,v in enumerate(votes)])/sum(votes) if sum(votes)>0 else 0

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
        print('length of choices=',len(choices))
        #debugging
        # print('**********debugging***********************')
        #print('selected choice', [node_id,data.node[node_id]['text']] )
        #print('indx=',indx,', choices=',choices)
        # print('choices_text=',choices_text) 
        # print('******************************************')

        
    return sentences, label, chosen_node_ids, extended_edges,sum(scores)/len(scores) if scores else 0

def one_d_random_graph_walk(sentences, data, node_id, child_edges, walk_len):

    print('generating new 1D graph walk')

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
      
        print('length of choices=',len(choices))

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



def visualize_graph_walk(tree_data, chosen_node_ids, extended_edges, start_node_id, label):
    # Initialize a directed graph
    G = nx.DiGraph()

    # Add nodes and edges from the entire tree
    for node_id in tree_data.node:
        G.add_node(node_id)
        
    for node_id in tree_data.edge.keys():
        edge= data.edge[node_id]
        if len(edge.keys()) > 0:
            G.add_edge(list(edge.keys())[0], node_id)  # for better visualization, the order is reversed

    # Create a layout for positioning the nodes
    pos = nx.spring_layout(G, k=0.3, seed=42)  # Increase `k` to spread nodes out, adjust `seed` as needed

    # Draw the full tree in a faded color (gray)
    nx.draw(G, pos, node_color='lightgray', edge_color='lightgray', with_labels=False,
            node_size=300, font_size=8, alpha=0.3)

    # Highlight the random walk nodes in blue
    nx.draw_networkx_nodes(G, pos, nodelist=[node for node in chosen_node_ids if node != start_node_id], node_color='blue', node_size=300, alpha=0.9)

    # Highlight the edges of the random walk in blue
    nx.draw_networkx_edges(G, pos, edgelist=[e for e in extended_edges if e[0] in chosen_node_ids and e[1] in chosen_node_ids],
                             edge_color='blue', width=1)

    # Highlight the start node in red
    nx.draw_networkx_nodes(G, pos, nodelist=[start_node_id], node_color='red', node_size=300, alpha=0.9)

    # Display the label (e.g., "informative") in a larger font at the top of the plot
    plt.figtext(0.5, 0.95, f"{label_map[label]}", ha="center", fontsize=22, fontweight='bold', color='purple')

    # Now, ensure the colored nodes and edges are drawn last by plotting them after the gray ones
    plt.axis('off')  # Turn off the axis for a cleaner look
    #plt.tight_layout()  # Adjust layout for better appearance

    # Define the full path of the folder and the file name
    folder_path = 'Kialo/Classification/random-walks/' + str(label)
    file_name = str(start_node_id) + '.jpg'
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name)
    plt.savefig(file_path, format="jpg", dpi=800)
    plt.savefig(os.path.join(folder_path, str(start_node_id) + '.pdf'), format="pdf", dpi=1200)


    # Show the plot for 1 second
    plt.show(block=False)  # Show plot without blocking further code execution
    plt.pause(3)           # Pause for 1 second to keep the plot visible
    plt.close()            # Close the plot window after 1 second
    
    #plt.show()





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


def plot_combined_cdf():
    """Plot combined CDF for all label categories."""
    plt.figure(figsize=(12, 6))
    for label in longest_paths.keys():
        if len(longest_paths[label]) >= population:
            data = sorted(longest_paths[label][:])
            cdf = np.arange(1, len(data) + 1) / len(data)
            plt.plot(data, cdf, label=label_map[label])

    plt.title('CDF of 2D Graph Walk Depth for Different Categories')
    plt.xlabel('Depth')
    plt.ylabel('Cumulative Density')
    plt.legend()


    # Set integer ticks on the x-axis
    #max_depth = max(max(longest_paths[label][:]) for label in longest_paths if len(longest_paths[label]) >= population)
    #plt.xticks(range(1, int(max_depth) + 1))

    cdf_file_path = 'Kialo/Classification/random-walks/combined_cdf.png'
    plt.savefig(cdf_file_path)
    plt.show()



# def plot_combined_cdf():
#     """Plot combined CDF for all label categories with different styles to reduce overlap."""
#     plt.figure(figsize=(12, 6))
    
#     # Define different line styles and markers for variety
#     line_styles = itertools.cycle(['-', '--', '-.', ':'])
#     markers = itertools.cycle(['o', 's', 'D', '^', 'v', 'P', '*'])

#     for label in longest_paths.keys():
#         if len(longest_paths[label]) >= population:
#             data = sorted(longest_paths[label][:])
#             cdf = np.arange(1, len(data) + 1) / len(data)
#             # Apply transparency (alpha), line style, and marker for each label
#             plt.plot(data, cdf, label=label, linestyle=next(line_styles), marker=next(markers), alpha=0.7)

#     plt.title('CDF of 2D Graph Walk Depth for Different Categories')
#     plt.xlabel('Depth')
#     plt.ylabel('Cumulative Density')
#     plt.legend()

#     # Set integer ticks on the x-axis
#     max_depth = max(max(longest_paths[label][:]) for label in longest_paths if len(longest_paths[label]) >= population)
#     plt.xticks(range(1, int(max_depth) + 1))

#     cdf_file_path = 'Kialo/Classification/random-walks/combined_cdf.png'
#     plt.savefig(cdf_file_path)
#     plt.show()


def plot_combined_pdf():
    """Plot combined PDF for all label categories with transparency to handle overlaps."""
    plt.figure(figsize=(12, 6))
    
    for label in longest_paths.keys():
        if len(longest_paths[label]) >= population:
            data = longest_paths[label][:]
            # Plot PDF using seaborn's KDE for smooth curves, or use plt.hist for a histogram
            sns.kdeplot(data, label=label_map[label], alpha=0.5, fill=True)

    plt.title('PDF of 2D Graph Walk Depth for Different Categories')
    plt.xlabel('Depth')
    plt.ylabel('Density')
    plt.legend()

    # Set integer ticks on the x-axis
    #max_depth = max(max(longest_paths[label][:]) for label in longest_paths if len(longest_paths[label]) >= population)
    #plt.xticks(range(1, int(max_depth) + 1))

    pdf_file_path = 'Kialo/Classification/random-walks/combined_pdf.png'
    plt.savefig(pdf_file_path)
    plt.show()





def plot_histogram(dic,name):
    """Plot combined PDF for all label categories with transparency to handle overlaps."""
    plt.figure(figsize=(12, 6))
    
    # Set the number of bins and the width of each histogram
    bins = 20
    bin_width = 0.1  # Adjust this value to control the spacing between histograms

    for idx, label in enumerate(dic.keys()):
        if len(dic[label]) >= population:
            data = dic[label][:]
            # Calculate the position for the histogram to offset each by its index
            bin_positions = np.linspace(min(data), max(data), bins)
            plt.hist(data, bins=bin_positions, density=True, alpha=0.5, 
                     label=label_map[label], align='mid', histtype='stepfilled')

    plt.title('PDF of 2D Graph Walk '+name+' for Different Categories')
    plt.xlabel(name)
    plt.ylabel('Probability Density')
    plt.legend()

    pdf_file_path = 'Kialo/Classification/random-walks/'+name+'_pdf.png'
    plt.savefig(pdf_file_path)
    plt.show()



def plot_pdf(dic,name):
    """Plot combined PDF for all label categories with transparency to handle overlaps."""
    plt.figure(figsize=(12, 6))
    
    for label in dic.keys():
        if len(dic[label]) >= population:
            data = dic[label][:]
            # Plot PDF using seaborn's KDE for smooth curves, or use plt.hist for a histogram
            sns.kdeplot(data, label=label_map[label], alpha=0.7, fill=True)

    #plt.title(name+' PDF of 2D Graph Walks Relative to 1D Graph Walks for Different Categories',fontsize=18)
    plt.xlabel(name,fontsize=18, fontweight='bold')
    plt.ylabel('Density',fontsize=18, fontweight='bold')
    plt.legend(fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Set integer ticks on the x-axis
    # max_depth = max(max(dic[label][:]) for label in dic if len(dic[label]) >= population)+2
    # min_depth = min(min(dic[label][:]) for label in dic if len(dic[label]) >= population)-2
    # plt.xticks(np.arange(min_depth, max_depth + 1, 0.5), fontsize=16)
    #plt.xticks(range(1, int(max_depth) + 1),fontsize=16)

    pdf_file_path = 'Kialo/Classification/random-walks/'+name+'_pdf.png'
    plt.savefig(pdf_file_path,dpi=800)

    pdf_file_path = 'Kialo/Classification/random-walks/' + name + '_pdf.pdf'
    plt.savefig(pdf_file_path, format='pdf', dpi=1200)
    #plt.show()

    plt.show(block=False)  # Show plot without blocking further code execution
    plt.pause(1)           # Pause for 1 second to keep the plot visible
    plt.close()            # Close the plot window after 1 second

def plot_pdf_all(all, name, pop=1000):
    plt.figure(figsize=(12, 6))

    # Select the data from the first 'population' elements of 'all'
    data = all[:]

    # Calculate KDE using scipy's gaussian_kde
    kde = gaussian_kde(data, bw_method=0.5)
    x_vals = np.linspace(min(data), max(data), 1000)
    y_vals = kde(x_vals)

    # Plot the KDE
    plt.plot(x_vals, y_vals, alpha=0.5)
    plt.fill_between(x_vals, y_vals, alpha=0.5)

    # Add vertical line at x = 1
    plt.axvline(x=1, color='red', linestyle='--', linewidth=1)

    # Conditional shading based on 'name'
    if 'Depth' in name:
        # Shade area to the right of x=1
        plt.fill_between(x_vals, 0, y_vals, where=(x_vals < 1), color='gray')
    elif 'Breadth' in name:
        # Shade area to the left of x=1
        plt.fill_between(x_vals, 0, y_vals, where=(x_vals < 1), color='gray')

    # Plot titles and labels
    #plt.title(name + ' PDF of 2D Graph Walks Relative to 1D Graph Walks', fontsize=18)
    plt.xlabel(name, fontsize=18, fontweight='bold')
    plt.ylabel('Density', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Save the plot as a PDF file
    pdf_file_path = 'Kialo/Classification/random-walks/' + name + '_all_pdf.png'
    plt.savefig(pdf_file_path, dpi=800)

    pdf_file_path = 'Kialo/Classification/random-walks/' + name + '_all_pdf.pdf'
    plt.savefig(pdf_file_path,format='pdf', dpi=1200)

    #plt.show()
    
    plt.show(block=False)  # Show plot without blocking further code execution
    plt.pause(1)           # Pause for 1 second to keep the plot visible
    plt.close()            # Close the plot window after 1 second


def plot_cdf(dic,name):
    """Plot combined CDF for all label categories."""
    plt.figure(figsize=(12, 6))
    for label in dic.keys():
        if len(dic[label]) >= population:
            data = sorted(dic[label][:])
            cdf = np.arange(1, len(data) + 1) / len(data)
            plt.plot(data, cdf, label=label_map[label])

    #plt.title(name+' CDF of 2D Graph Walks Relative to 1D Graph Walks for Different Categories',fontsize=18)
    plt.xlabel(name,fontsize=18, fontweight='bold')
    plt.ylabel('Cumulative Density',fontsize=18, fontweight='bold')
    plt.legend(fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Set integer ticks on the x-axis
    #max_depth = max(max(longest_paths[label][:]) for label in longest_paths if len(longest_paths[label]) >= population)
    #plt.xticks(range(1, int(max_depth) + 1))

    cdf_file_path = 'Kialo/Classification/random-walks/'+name+'_cdf.png'
    plt.savefig(cdf_file_path,dpi=800)

    cdf_file_path = 'Kialo/Classification/random-walks/'+name+'_cdf.pdf'
    plt.savefig(cdf_file_path,format='pdf',dpi=1200)

    #plt.show()
    
    plt.show(block=False)  # Show plot without blocking further code execution
    plt.pause(1)           # Pause for 1 second to keep the plot visible
    plt.close()            # Close the plot window after 1 second

def plot_interpolated_cdf(dic, name, num_interp_points=500):
    """Plot combined, smoothed CDF for all label categories with interpolation for clear visibility."""
    plt.figure(figsize=(12, 6))

    for idx, (label, values) in enumerate(dic.items()):
        if len(values) >= population:
            # Step 1: Sort and limit the data
            data = sorted(values[:])
            cdf = np.arange(1, len(data) + 1) / len(data)
            
            # Optionally, remove duplicates from data and corresponding CDF values
            data, indices = np.unique(data, return_index=True)
            cdf = cdf[indices]

            # Step 2: Interpolate the CDF to get a smoother curve
            interpolator = interp1d(data, cdf, kind='quadratic')
            interpolated_data = np.linspace(data[0], data[-1], num_interp_points)
            interpolated_cdf = interpolator(interpolated_data)
            
            # Step 3: Plot with slight offset in alpha and line style
            plt.plot(interpolated_data, interpolated_cdf, label=label_map[label])
    
    #plt.title(f"{name} Interpolated CDF of 2D Graph Walks Relative to 1D Graph Walks for Different Categories",fontsize=18)
    plt.xlabel(name,fontsize=18, fontweight='bold')
    plt.ylabel('Cumulative Density',fontsize=18, fontweight='bold')
    plt.legend(fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Save and show the figure
    cdf_file_path = f'Kialo/Classification/random-walks/{name}_interpolated_cdf.png'
    plt.savefig(cdf_file_path, dpi=800)

    cdf_file_path = f'Kialo/Classification/random-walks/{name}_interpolated_cdf.pdf'
    plt.savefig(cdf_file_path, format='pdf',dpi=1200)

    #plt.show()
    
    plt.show(block=False)  # Show plot without blocking further code execution
    plt.pause(1)           # Pause for 1 second to keep the plot visible
    plt.close()            # Close the plot window after 1 second




def plot_comparison_bar():
    """Plot a bar plot comparing average scores for each label in relevance_scores (2D Walk) and relevance_scores1 (1D Walk)."""
    labels = []
    avg_scores_2d = []
    avg_scores_1d = []
    
    # Calculate average scores for each label if population threshold is met
    for label in relevance_scores.keys():
        if len(relevance_scores[label]) >= population and len(relevance_scores1[label]) >= population:
            labels.append(label_map[label])
            avg_scores_2d.append(sum(relevance_scores[label][:]) / len(relevance_scores[label][:]))
            avg_scores_1d.append(sum(relevance_scores1[label][:]) / len(relevance_scores1[label][:]))

    # Dynamically adjust spacing and bar width
    num_labels = len(labels)
    spacing_factor = 1.5
    x = np.arange(num_labels) * spacing_factor  # Label locations on the x-axis
    width = 0.3 / spacing_factor  # Adjust bar width based on spacing

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))  # Adjust figure size
    bars_2d = ax.bar(x - width/2, avg_scores_2d, width, label='2D Walk', color='skyblue')
    bars_1d = ax.bar(x + width/2, avg_scores_1d, width, label='1D Walk', color='salmon')


    # Adding labels, title, and x-tick labels for each label
    #ax.set_xlabel('Labels', fontsize=18)
    ax.set_ylabel('Relevance Score', fontsize=18, fontweight='bold')
    #ax.set_title('Comparison of Semantic Relevance Scores between 2D and 1D Walks by category', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=18,fontweight='bold')
    ax.legend(fontsize=16,loc='upper center')
    plt.yticks(fontsize=16)
    plt.tight_layout()

    bar_file_path = f'Kialo/Classification/random-walks/scores_bar_plot.png'
    plt.savefig(bar_file_path, dpi=800)

    bar_file_path = f'Kialo/Classification/random-walks/scores_bar_plot.pdf'
    plt.savefig(bar_file_path, format='pdf', dpi=1200)

    # Display the plot
    #plt.show()
    plt.show(block=False)  # Show plot without blocking further code execution
    plt.pause(1)           # Pause for 1 second to keep the plot visible
    plt.close()            # Close the plot window after 1 second

def plot_average_values_categorical(dic, name):
    """
    Plot a 2D bar chart of average values for categorical labels (e.g., "Attacking" or "Supporting").
    """
    plt.figure(figsize=(6, 6))
    
    # Extract averages for each category
    categories = []
    averages = []
    for label, values in dic.items():
        if len(values) >= population:  # Ensure there's enough data in the category
            categories.append(label_map[label])  # Label as a category (e.g., "Attacking" or "Supporting")
            averages.append(np.mean(values[:]))  # Compute average
    
    color_palette = ['#FFC107', '#708090'] 
    category_colors = {category: color_palette[i % len(color_palette)] for i, category in enumerate(categories)}
    
    # Define the bar width
    bar_width = 0.3  # Set a smaller width for thinner bars
    
    # Plot the averages as a bar chart
    bar_colors = [category_colors[category] for category in categories]
    plt.bar(categories, averages, width=bar_width, color=bar_colors, alpha=0.8, edgecolor='black')
    
    # Customize the plot
    #plt.xlabel("Category", fontsize=18, fontweight='bold')
    plt.ylabel(f"{name}", fontsize=16, fontweight='bold')
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16)

    plt.tight_layout()
    
    # Save the plot as PNG and PDF
    avg_file_path_png = f'Kialo/Classification/random-walks/{name}_average_values.png'
    plt.savefig(avg_file_path_png, dpi=800)

    avg_file_path_pdf = f'Kialo/Classification/random-walks/{name}_average_values.pdf'
    plt.savefig(avg_file_path_pdf, format='pdf', dpi=1200)

    # Display the plot
    plt.show(block=False)  # Show plot without blocking further code execution
    plt.pause(1)           # Pause for 1 second to keep the plot visible
    plt.close()            # Close the plot window after 1 second

def plot_whisker_all(data, name):
    plt.figure(figsize=(6, 8))

    # Statistics
    mean_val = np.mean(data)
    median_val = np.median(data)
    p25 = np.percentile(data, 25)
    p75 = np.percentile(data, 75)

    # Create boxplot
    plt.boxplot(data,
                vert=True,
                patch_artist=True,
                widths=0.4,
                boxprops=dict(facecolor='lightgray', color='black', linewidth=1.5),
                whiskerprops=dict(color='black', linewidth=1.5),
                capprops=dict(color='black', linewidth=1.5),
                medianprops=dict(color='black', linewidth=0),  # We'll draw it ourselves
                flierprops=dict(marker='o', markerfacecolor='black', markersize=4, linestyle='none'))

    # Mean (solid red)
    plt.axhline(y=mean_val, color='red', linestyle='-', linewidth=1.5,  label=f'Mean = {mean_val:.2f}')

    # Median (dashed black)
    plt.axhline(y=median_val, color='black', linestyle='--', linewidth=1.2, label=f'Median = {median_val:.2f}')

    # Percentiles (faint gray)
    # plt.axhline(y=p25, color='gray', linestyle=':', linewidth=1)
    # plt.axhline(y=p75, color='gray', linestyle=':', linewidth=1)
    # plt.text(1.1, p25, '25th %ile', fontsize=12)
    # plt.text(1.1, p75, '75th %ile', fontsize=12)

    # Set ticks manually
    max_y = max(max(data), 1)
    y_ticks = np.arange(0, max_y + 0.2, 0.5)
    plt.yticks(y_ticks, fontsize=16)
    plt.xticks([])  # Remove x-axis ticks

    # Y-axis label
    plt.ylabel(name, fontsize=18, fontweight='bold')

    # Remove all spines and x-axis
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_visible(False)  # Fully remove x-axis

    # Add legend
    plt.legend(loc='upper right', fontsize=14, frameon=False)

    # Final formatting and export
    plt.tight_layout()
    plt.savefig(f'Kialo/Classification/random-walks/{name}_boxplot.png', dpi=800)
    plt.savefig(f'Kialo/Classification/random-walks/{name}_boxplot.pdf', format='pdf', dpi=1200)

    plt.show(block=False)
    plt.pause(1)
    plt.close()


def plot_two_whiskers(data1, name1, data2, name2):
    # Extract labels
    xtick1 = name1.split('(')[0].strip() + ' (2D/1D)'
    xtick2 = name2.split('(')[0].strip() + ' (2D/1D)'

    # Main colors
    color2 = '#C38081'     # maroon box
    color1 = '#80c3c2'     # turquoise box
    outlier_edge2 = '#994949'  # deeper red
    outlier_edge1 = '#3b8f8e'  # deeper teal
    mean_color2 = 'red'
    mean_color1 = '#19807e'
    median_color2 = 'red'
    median_color1 = '#19807e'

    # Convert data
    data1 = np.array(data1, dtype=np.float64)
    data2 = np.array(data2, dtype=np.float64)
    data = [data1, data2]

    plt.figure(figsize=(9.5, 8))

    # --- Baseline at y = 1.0, behind everything ---
    plt.axhline(y=1.0, color='black', linewidth=2, zorder=1)

    # --- Boxplot ---
    bp = plt.boxplot(data,
                     vert=True,
                     patch_artist=True,
                     widths=0.4,
                     boxprops=dict(facecolor=color1, color=color1, linewidth=1.5, zorder=3),
                     whiskerprops=dict(color=color1, linewidth=1.5, zorder=3),
                     capprops=dict(color=color1, linewidth=1.5, zorder=3),
                     medianprops=dict(color=median_color1, linewidth=3, linestyle='--', zorder=4),
                     flierprops=dict(marker='o', markerfacecolor=color1, markeredgecolor=outlier_edge1,
                                     markersize=4, linestyle='none', zorder=3))

    # Update 2nd box (Breadth)
    for patch in [bp['boxes'][1], bp['whiskers'][2], bp['whiskers'][3],
                  bp['caps'][2], bp['caps'][3]]:
        patch.set_color(color2)
        if hasattr(patch, 'set_facecolor'):
            patch.set_facecolor(color2)
    bp['medians'][1].set_color(median_color2)
    bp['medians'][1].set_linestyle('--')
    bp['medians'][1].set_linewidth(2)

    # Fix outliers
    for i, flier in enumerate(bp['fliers']):
        if i == 1:
            flier.set(markerfacecolor=color2, markeredgecolor=outlier_edge2)
        else:
            flier.set(markerfacecolor=color1, markeredgecolor=outlier_edge1)

    # Compute stats
    mean1, mean2 = np.mean(data1), np.mean(data2)
    median1, median2 = np.median(data1), np.median(data2)

    # --- Draw mean lines inside box boundaries, like medians ---
    box1_path = bp['boxes'][0].get_path().vertices
    box1_left = box1_path[:, 0].min()
    box1_right = box1_path[:, 0].max()

    box2_path = bp['boxes'][1].get_path().vertices
    box2_left = box2_path[:, 0].min()
    box2_right = box2_path[:, 0].max()

    plt.hlines(y=mean1, xmin=box1_left, xmax=box1_right, color=mean_color1, linestyle='-', linewidth=2, zorder=5)
    plt.hlines(y=mean2, xmin=box2_left, xmax=box2_right, color=mean_color2, linestyle='-', linewidth=2, zorder=5)

    # Get y-position just above upper whiskers for annotation
    upper_whisker1 = bp['whiskers'][1].get_ydata()[1]
    upper_whisker2 = bp['whiskers'][3].get_ydata()[1]
    offset = 0.1

    # Annotate Depth box (x=1.15)
    plt.text(1.15, upper_whisker2 + offset + 0.4, rf'$\mathbf{{───}}$  Mean = {mean1:.2f}',
             color=mean_color1, fontsize=16, verticalalignment='bottom')
    plt.text(1.15, upper_whisker2 + offset, rf'$\mathbf{{--}}$ Median = {median1:.2f}',
             color=median_color1, fontsize=16, verticalalignment='bottom')

    # Annotate Breadth box (x=2.15)
    plt.text(2.15, upper_whisker2 + offset + 0.4, rf'$\mathbf{{───}}$  Mean = {mean2:.2f}',
             color=mean_color2, fontsize=16, verticalalignment='bottom')
    plt.text(2.15, upper_whisker2 + offset, rf'$\mathbf{{--}}$ Median = {median2:.2f}',
             color=median_color2, fontsize=16, verticalalignment='bottom')

    # Plot styling
    max_y = max(np.max(data1), np.max(data2), 1)
    y_ticks = np.arange(0, max_y + 0.2, 0.5)
    plt.yticks(y_ticks, fontsize=14)
    plt.xticks([1, 2], [xtick1, xtick2], fontsize=18, fontweight='bold')

    # Remove x-axis tick marks
    ax = plt.gca()
    ax.tick_params(axis='x', length=0)
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)

    plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)

    # Save and display
    plt.tight_layout()
    plt.savefig(f'Kialo/Classification/random-walks/2D_vs_1D_boxplot_g.png', dpi=800)
    plt.savefig(f'Kialo/Classification/random-walks/2D_vs_1D_boxplot_g.pdf', format='pdf', dpi=1200)
    plt.show(block=False)
    plt.pause(1)
    plt.close()





print("Current working directory:", os.getcwd())

# Split dataset into train and test set.
dataset_path = 'Kialo/Classification/random-walks/serializedGraphs/' #  or provide the dataset path C:\Users\za0005\OneDrive - University of Surrey\Desktop\Surrey\NLI\Datasets\kialo_debates\serializedGraphs
files = os.listdir(dataset_path)
dataset_samples = []
dataset_samples1=[]
labels = []

model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)


for indx, file in enumerate(files):
    if indx >= 0:
        print('Processing', indx, file)
        data = pkl.load(open(dataset_path + file, 'rb'))

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
            print('Processing', indx, file)

            #2D
            sentences = ['']*walk_length*3
            sentences, label,chosen_node_ids, extended_edges,score = random_graph_walk(sentences, data, node_id, child_edges, walk_length)
            #1D
            sentences1 = ['']*walk_length*3
            sentences1, label1,chosen_node_ids1, extended_edges1,score1=one_d_random_graph_walk(sentences1, data, node_id, child_edges, walk_length)
            
            

            if label != -1 and label1!='' and not sentences[2]=='' and not sentences1[2]=='' and not sentences[5]=='' and not sentences1[5]=='':
                sentences.append(label)
                sentences1.append(label1) #1D
                dataset_samples.append(sentences)
                dataset_samples1.append(sentences1)
                print('sample',len(dataset_samples),'added with label:',label)

                #relevance scores
                relevance_scores[label].append(score)
                relevance_scores1[label].append(score1)

                #Depth metric
                longest_path_length = calculate_longest_path(extended_edges, chosen_node_ids, label,[chosen_node_ids[0]])
                longest_path_length1 = calculate_longest_path(extended_edges1, chosen_node_ids1, label1,[chosen_node_ids1[0]])
                depth_ratio=longest_path_length/longest_path_length1
                longest_paths[label].append(depth_ratio)
                longest_paths_all.append(depth_ratio)

                #mean branching metric
                out_degree_count = defaultdict(int)
                for parent, child in extended_edges:
                    if parent in chosen_node_ids and child in chosen_node_ids:
                        out_degree_count[parent] += 1
                mean_branching_factor_walk = sum(out_degree_count.values())/len(out_degree_count)

                out_degree_count = defaultdict(int)
                for parent, child in extended_edges1:
                    if parent in chosen_node_ids1 and child in chosen_node_ids1:
                        out_degree_count[parent] += 1
                mean_branching_factor_walk1 = sum(out_degree_count.values())/len(out_degree_count)

                mean_branching_ratio=mean_branching_factor_walk/mean_branching_factor_walk1
                mean_branching_factors[label].append(mean_branching_ratio)
                mean_branching_factors_all.append(mean_branching_ratio)

                print('a mean branching factor for a 2D walk=',mean_branching_factor_walk)
                print('a mean branching factor for a 1D walk=',mean_branching_factor_walk1)

                if len(dataset_samples)==plot_dist_after and plot_early:
                    
                    plot_comparison_bar()

                    plot_pdf(longest_paths,'Relative Depth (2D vs 1D Walk)')
                    plot_pdf_all(longest_paths_all,'Relative Depth (2D vs 1D Walk)',len(longest_paths_all))
                    plot_cdf(longest_paths,'Relative Depth (2D vs 1D Walk)')
                    plot_interpolated_cdf(longest_paths,'Relative Depth (2D vs 1D Walk)')
                    plot_average_values_categorical(longest_paths,'Relative Depth (2D vs 1D Walk)')
                    plot_whisker_all(longest_paths_all,'Relative Depth (2D vs 1D Walk)')

                    plot_pdf(mean_branching_factors,'Relative Breadth (2D vs 1D Walk)')
                    plot_pdf_all(mean_branching_factors_all,'Relative Breadth (2D vs 1D Walk)',len(mean_branching_factors_all))
                    plot_cdf(mean_branching_factors,'Relative Breadth (2D vs 1D Walk)')
                    plot_interpolated_cdf(mean_branching_factors,'Relative Breadth (2D vs 1D Walk)')
                    plot_average_values_categorical(mean_branching_factors,'Relative Breadth (2D vs 1D Walk)')
                    plot_whisker_all(mean_branching_factors_all,'Relative Breadth (2D vs 1D Walk)')

                    plot_two_whiskers(longest_paths_all,'Relative Depth (2D vs 1D Walk)',mean_branching_factors_all,'Relative Breadth (2D vs 1D Walk)' )

                    sys.exit()
                    


                #visualization
                #visualize_graph_walk(data, chosen_node_ids, extended_edges, node_id, label)
                #visualize_graph_walk(data, chosen_node_ids1, extended_edges1, node_id, label1)

if plot_end:
    plot_comparison_bar()

    plot_pdf(longest_paths,'Relative Depth (2D vs 1D Walk)')
    plot_pdf_all(longest_paths_all,'Relative Depth (2D vs 1D Walk)',len(longest_paths_all))
    plot_cdf(longest_paths,'Relative Depth (2D vs 1D Walk)')
    plot_interpolated_cdf(longest_paths,'Relative Depth (2D vs 1D Walk)')
    plot_average_values_categorical(longest_paths,'Relative Depth (2D vs 1D Walk)')
    plot_whisker_all(longest_paths_all,'Relative Depth (2D vs 1D Walk)')

    plot_pdf(mean_branching_factors,'Relative Breadth (2D vs 1D Walk)')
    plot_pdf_all(mean_branching_factors_all,'Relative Breadth (2D vs 1D Walk)',len(mean_branching_factors_all))
    plot_cdf(mean_branching_factors,'Relative Breadth (2D vs 1D Walk)')
    plot_interpolated_cdf(mean_branching_factors,'Relative Breadth (2D vs 1D Walk)')
    plot_average_values_categorical(mean_branching_factors,'Relative Breadth (2D vs 1D Walk)')
    plot_whisker_all(mean_branching_factors_all,'Relative Breadth (2D vs 1D Walk)')

    plot_two_whiskers(longest_paths_all,'Relative Depth (2D vs 1D Walk)',mean_branching_factors_all,'Relative Breadth (2D vs 1D Walk)' )

if not save_data:
    sys.exit()


print('#samples 2D:', len(dataset_samples))
random.shuffle(dataset_samples)

train_samples = dataset_samples[ : math.ceil(0.8*len(dataset_samples))]
dev_samples = dataset_samples[math.ceil(0.8*len(dataset_samples)) : ]
cols = []
for i in range(1, walk_length + 1):
    cols.append(f'parent_id{i}')  # Parent ID for each sentence
    cols.append(f'id{i}')        # ID for each sentence
    cols.append(f'sent{i}')      # Sentence
cols.append('label')
#pd.DataFrame(train_samples, columns=cols).to_csv('Kialo/Classification/random-walks/GG_train_simil_random_walk.csv', index=False)
#pd.DataFrame(dev_samples, columns=cols).to_csv('Kialo/Classification/random-walks/GG_test_simil_random_walk.csv', index=False)
pd.DataFrame(train_samples, columns=cols).to_csv('Kialo/Classification/random-walks/GK_train_random_walk.csv', index=False)
pd.DataFrame(dev_samples, columns=cols).to_csv('Kialo/Classification/random-walks/GK_test_random_walk.csv', index=False)

print('#train samples 2D:', len(train_samples))
print('#test samples 2D:', len(dev_samples))


print('#samples 1D:', len(dataset_samples1))
random.shuffle(dataset_samples1)

train_samples1 = dataset_samples1[ : math.ceil(0.8*len(dataset_samples1))]
dev_samples1 = dataset_samples1[math.ceil(0.8*len(dataset_samples1)) : ]
cols = []
for i in range(1, walk_length + 1):
    cols.append(f'parent_id{i}')  # Parent ID for each sentence
    cols.append(f'id{i}')        # ID for each sentence
    cols.append(f'sent{i}')      # Sentence
cols.append('label')
# pd.DataFrame(train_samples1, columns=cols).to_csv('Kialo/Classification/random-walks/one_GG_train_simil_random_walk.csv', index=False)
# pd.DataFrame(dev_samples1, columns=cols).to_csv('Kialo/Classification/random-walks/one_GG_test_simil_random_walk.csv', index=False)
pd.DataFrame(train_samples1, columns=cols).to_csv('Kialo/Classification/random-walks/one_GK_train_random_walk.csv', index=False)
pd.DataFrame(dev_samples1, columns=cols).to_csv('Kialo/Classification/random-walks/one_GK_test_random_walk.csv', index=False)


print('#train samples 1D:', len(train_samples1))
print('#test samples 1D:', len(dev_samples1))
