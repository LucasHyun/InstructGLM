import numpy as np
import networkx as nx
import random
from sentence_transformers import SentenceTransformer

# Initialize the similarity model (SimCSE or similar)
similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def compute_similarity(text1, text2):
    """
    Compute cosine similarity between two text embeddings.
    """
    embedding1 = similarity_model.encode(text1)
    embedding2 = similarity_model.encode(text2)
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def pinsage_biased_sampling(graph, node_features, budget, hops=2, top_k=3):
    """
    Combine PinSAGE with Similarity-based Neighbor Selection (SNS).

    Parameters:
        graph (nx.Graph): Graph structure.
        node_features (dict): Node features as text attributes. {node: text_feature}
        budget (int): Number of nodes to sample.
        hops (int): Number of hops for graph traversal.
        top_k (int): Number of top neighbors based on similarity.

    Returns:
        dict: Generated prompts with sampled nodes.
    """
    prompts = {}
    for node in graph.nodes:
        neighbors = nx.single_source_shortest_path_length(graph, node, cutoff=hops)
        neighbors = [n for n in neighbors if n != node]
        
        # Rank neighbors by similarity
        similarities = [(neighbor, compute_similarity(node_features[node], node_features[neighbor])) 
                        for neighbor in neighbors]
        top_neighbors = sorted(similarities, key=lambda x: -x[1])[:top_k]
        
        # Generate prompt
        prompt = f"Node {node} has features: {node_features[node]}\n"
        prompt += "Top similar neighbors:\n"
        for rank, (neighbor, sim) in enumerate(top_neighbors, 1):
            prompt += f"{rank}. Node {neighbor} (similarity: {sim:.2f}): {node_features[neighbor]}\n"
        
        prompts[node] = prompt
    
    return prompts

# Example Usage
if __name__ == "__main__":
    # Create a sample graph
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (1, 3)])
    
    # Define node features
    node_features = {
        0: "Deep learning for image classification.",
        1: "Convolutional networks in computer vision.",
        2: "Reinforcement learning for robotics.",
        3: "Applications of NLP in healthcare.",
        4: "Graph neural networks for node classification."
    }
    
    # Generate prompts
    budget = 3
    prompts = pinsage_biased_sampling(graph, node_features, budget)
    for node, prompt in prompts.items():
        print(f"Prompt for Node {node}:\n{prompt}\n")
