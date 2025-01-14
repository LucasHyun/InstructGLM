import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import numpy as np
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing


class PinSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_neighbors):
        super(PinSAGE, self).__init__()
        self.num_neighbors = num_neighbors
        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Neighborhood aggregation
        row, col = edge_index
        out = torch.zeros_like(x)
        for node in range(x.size(0)):
            neighbors = col[row == node]
            sampled_neighbors = neighbors[:self.num_neighbors]  # Sampling neighbors
            neighbor_features = x[sampled_neighbors]
            out[node] = neighbor_features.mean(dim=0) if len(neighbor_features) > 0 else x[node]

        # Linear transformation
        out = self.linear(out)
        return F.relu(out)

class PinSAGE(torch.nn.Module):
    """
    Simple PinSAGE-like implementation for node importance sampling.
    """
    def __init__(self, in_channels, out_channels, num_neighbors):
        super(PinSAGE, self).__init__()
        self.num_neighbors = num_neighbors
        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Neighborhood aggregation
        row, col = edge_index
        out = torch.zeros_like(x)
        for node in range(x.size(0)):
            neighbors = col[row == node]
            sampled_neighbors = neighbors[:self.num_neighbors]  # Sampling neighbors
            neighbor_features = x[sampled_neighbors]
            out[node] = neighbor_features.mean(dim=0) if len(neighbor_features) > 0 else x[node]

        # Linear transformation
        out = self.linear(out)
        return out

def compute_similarity(text1, text2, similarity_model):
    """
    Compute cosine similarity between two text embeddings.
    """
    embedding1 = similarity_model.encode(text1)
    embedding2 = similarity_model.encode(text2)
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def sns_rank_neighbors(node, graph, features, similarity_model, top_k=3):
    """
    Rank neighbors based on similarity scores using SNS.
    """
    neighbors = graph.edge_index[1][graph.edge_index[0] == node].tolist()
    node_text = f"Node {node} has features {features[node].tolist()}"
    neighbor_scores = []

    for neighbor in neighbors:
        neighbor_text = f"Node {neighbor} has features {features[neighbor].tolist()}"
        score = compute_similarity(node_text, neighbor_text, similarity_model)
        neighbor_scores.append((neighbor, score))

    ranked_neighbors = sorted(neighbor_scores, key=lambda x: -x[1])[:top_k]
    return [n[0] for n in ranked_neighbors]

def generate_prompt_for_node(node, graph, features, similarity_model, top_k=3):
    """
    Generate enhanced prompts for a given node.
    """
    top_neighbors = sns_rank_neighbors(node, graph, features, similarity_model, top_k=top_k)
    feature_desc = f"Node {node} has features {features[node].tolist()}.\n"
    connections = " ".join([f"Node {node} is connected to Node {n}." for n in top_neighbors])
    return feature_desc + connections

def main():
    # Example graph and features
    node_features = torch.rand(10, 5)  # 10 nodes with 5 features each
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8],
                                [1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
    graph = Data(x=node_features, edge_index=edge_index)

    # Initialize PinSAGE
    pinsage = PinSAGE(in_channels=node_features.size(1), out_channels=node_features.size(1), num_neighbors=3)
    sampled_embeddings = pinsage(node_features, graph.edge_index)

    # Initialize SentenceTransformer for SNS
    similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Generate prompts for sampled nodes
    for node in range(node_features.size(0)):
        prompt = generate_prompt_for_node(node, graph, node_features, similarity_model, top_k=3)
        print(f"Prompt for Node {node}:\n{prompt}\n")

if __name__ == "__main__":
    main()
