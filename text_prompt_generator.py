import numpy as np
import random

def biased_sampling(graph, features, budget, reward_function, temperature=1.0, hops=2):
    """
    Biased sampling for graph-based prompts in InstructGLM.

    Parameters:
        graph (dict): Adjacency list representing the graph. {node: [neighbors]}
        features (dict): Node features. {node: feature_vector}
        budget (int): Number of nodes to sample.
        reward_function (callable): Function to compute reward for a node.
        temperature (float): Temperature parameter for sampling.
        hops (int): Number of hops to include in the textual prompt.

    Returns:
        list: Generated prompts for the sampled nodes.
    """
    nodes = list(graph.keys())
    probabilities = {node: 1 / len(nodes) for node in nodes}  # Initialize equal probabilities
    sampled_nodes = []

    for _ in range(budget):
        # Sample a node based on probabilities
        sampled_node = random.choices(nodes, weights=[probabilities[node] for node in nodes])[0]
        sampled_nodes.append(sampled_node)

        # Compute reward for the sampled node
        reward = reward_function(sampled_node, graph, features)

        # Update probabilities using softmax with temperature
        probabilities = {
            node: np.exp(reward / temperature) / sum(np.exp(reward / temperature) for reward in 
                       [reward_function(n, graph, features) for n in nodes])
            for node in nodes
        }

    # Generate textual prompts for sampled nodes
    prompts = []
    for node in sampled_nodes:
        prompt = generate_text_prompt(node, graph, features, hops)
        prompts.append(prompt)

    return prompts


def generate_text_prompt(node, graph, features, hops):
    """
    Generate a natural language prompt for a given node.

    Parameters:
        node (int): Node ID.
        graph (dict): Adjacency list representing the graph.
        features (dict): Node features.
        hops (int): Number of hops to include in the description.

    Returns:
        str: Textual description of the graph centered around the node.
    """
    neighbors = get_neighbors(node, graph, hops)
    feature_desc = f"Node {node} has features {features[node]}.\n"
    connections = " ".join([f"Node {node} is connected to Node {n}." for n in neighbors])
    return feature_desc + connections


def get_neighbors(node, graph, hops):
    """
    Retrieve neighbors up to a specified number of hops.

    Parameters:
        node (int): Node ID.
        graph (dict): Adjacency list representing the graph.
        hops (int): Number of hops.

    Returns:
        list: List of neighbors within the specified hops.
    """
    visited = set()
    to_visit = [node]
    for _ in range(hops):
        next_level = []
        for current in to_visit:
            visited.add(current)
            next_level.extend([neighbor for neighbor in graph[current] if neighbor not in visited])
        to_visit = next_level
    return visited - {node}


# Example reward function (based on degree centrality)
def reward_function(node, graph, features):
    return len(graph[node])  # Reward based on the number of neighbors (degree centrality)


# Example usage
if __name__ == "__main__":
    # Example graph and features
    example_graph = {
        0: [1, 2],
        1: [0, 2, 3],
        2: [0, 1, 4],
        3: [1],
        4: [2]
    }
    example_features = {
        0: [0.1, 0.2],
        1: [0.3, 0.4],
        2: [0.5, 0.6],
        3: [0.7, 0.8],
        4: [0.9, 1.0]
    }

    # Run biased sampling
    prompts = biased_sampling(example_graph, example_features, budget=3, reward_function=reward_function)
    for prompt in prompts:
        print(prompt)
