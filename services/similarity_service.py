import numpy as np
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from models.agent import Agent, AgentCard


class GNNSimilarityModel(torch.nn.Module):
    """Graph Neural Network model for agent similarity."""

    def __init__(self, num_features: int, hidden_channels: int, embedding_dim: int):
        super(GNNSimilarityModel, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, embedding_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


class SimilarityService:
    """Service for agent similarity search using GNN."""

    def __init__(self, db: Session):
        self.db = db
        self.graph = None
        self.embeddings = None
        self.agent_ids = None
        self.model = None
        self.initialized = False

    def _initialize_model(self, num_features: int):
        """Initialize the GNN model."""
        hidden_channels = 64
        embedding_dim = 32
        self.model = GNNSimilarityModel(num_features, hidden_channels, embedding_dim)

    def _create_agent_features(self, agents: List[Agent]) -> Tuple[torch.Tensor, List[int]]:
        """Create feature vectors for agents."""
        features = []
        agent_ids = []

        # Get all unique capabilities and tags
        all_capabilities = set()
        all_tags = set()

        for agent in agents:
            if agent.capabilities:
                all_capabilities.update(agent.capabilities)
            
            # Get agent card tags
            agent_card = self.db.query(AgentCard).filter(AgentCard.agent_id == agent.id).first()
            if agent_card and agent_card.tags:
                all_tags.update(agent_card.tags)

        # Create a mapping from capability/tag to index
        capability_to_idx = {cap: i for i, cap in enumerate(all_capabilities)}
        tag_to_idx = {tag: i + len(capability_to_idx) for i, tag in enumerate(all_tags)}
        
        # Feature vector size: capabilities + tags + agent_type one-hot
        agent_types = set(agent.agent_type for agent in agents)
        agent_type_to_idx = {t: i for i, t in enumerate(agent_types)}
        feature_size = len(capability_to_idx) + len(tag_to_idx) + len(agent_type_to_idx)

        # Create feature vectors
        for agent in agents:
            feature = np.zeros(feature_size)
            
            # Set capability features
            if agent.capabilities:
                for cap in agent.capabilities:
                    if cap in capability_to_idx:
                        feature[capability_to_idx[cap]] = 1
            
            # Set tag features
            agent_card = self.db.query(AgentCard).filter(AgentCard.agent_id == agent.id).first()
            if agent_card and agent_card.tags:
                for tag in agent_card.tags:
                    if tag in tag_to_idx:
                        feature[tag_to_idx[tag]] = 1
            
            # Set agent type one-hot
            type_idx = len(capability_to_idx) + len(tag_to_idx) + agent_type_to_idx[agent.agent_type]
            feature[type_idx] = 1
            
            features.append(feature)
            agent_ids.append(agent.id)

        return torch.tensor(features, dtype=torch.float), agent_ids

    def _create_agent_graph(self, agents: List[Agent]) -> nx.Graph:
        """Create a graph of agents based on collaborations."""
        graph = nx.Graph()
        
        # Add nodes
        for i, agent in enumerate(agents):
            graph.add_node(i, agent_id=agent.id)
        
        # Add edges based on collaborations
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents):
                if i != j:
                    # Check if there's a collaboration between these agents
                    from models.collaboration import Collaboration
                    
                    collaboration = self.db.query(Collaboration).filter(
                        ((Collaboration.initiator_agent_id == agent1.id) & 
                         (Collaboration.collaborator_agent_id == agent2.id)) |
                        ((Collaboration.initiator_agent_id == agent2.id) & 
                         (Collaboration.collaborator_agent_id == agent1.id))
                    ).first()
                    
                    if collaboration and collaboration.status in ["active", "completed"]:
                        graph.add_edge(i, j)
        
        return graph

    def _convert_to_pytorch_geometric(self, graph: nx.Graph, features: torch.Tensor) -> Data:
        """Convert NetworkX graph to PyTorch Geometric data."""
        edge_index = []
        for edge in graph.edges():
            # Add both directions for undirected graph
            edge_index.append([edge[0], edge[1]])
            edge_index.append([edge[1], edge[0]])  # Reverse direction
        
        if not edge_index:  # If no edges, create a self-loop for each node
            edge_index = [[i, i] for i in range(len(graph.nodes()))] * 2
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        return Data(x=features, edge_index=edge_index)

    def build_similarity_model(self):
        """Build the similarity model from the current agents."""
        # Get all agents
        agents = self.db.query(Agent).all()
        if not agents:
            self.initialized = False
            return
        
        # Create feature vectors
        features, agent_ids = self._create_agent_features(agents)
        
        # Create agent graph
        graph = self._create_agent_graph(agents)
        
        # Convert to PyTorch Geometric format
        data = self._convert_to_pytorch_geometric(graph, features)
        
        # Initialize model if needed
        if self.model is None or features.shape[1] != self.model.conv1.in_channels:
            self._initialize_model(features.shape[1])
        
        # Generate embeddings
        self.model.eval()
        with torch.no_grad():
            self.embeddings = self.model(data.x, data.edge_index)
        
        self.agent_ids = agent_ids
        self.initialized = True

    def find_similar_agents(self, agent_id: str, top_k: int = 5, similarity_threshold: float = 0.5) -> List[Dict]:
        """Find similar agents to the given agent using the GNN model."""
        if not self.initialized:
            self.build_similarity_model()
            if not self.initialized:
                return []
        # Find the index of the agent
        if agent_id not in self.agent_ids:
            return []
        agent_idx = self.agent_ids.index(agent_id)
        # Get the embedding for the agent
        agent_embedding = self.embeddings[agent_idx].unsqueeze(0)
        # Calculate cosine similarity with all other agents
        similarities = []
        for i, other_id in enumerate(self.agent_ids):
            if other_id != agent_id:
                other_embedding = self.embeddings[i].unsqueeze(0)
                similarity = cosine_similarity(agent_embedding.detach().numpy(), other_embedding.detach().numpy())[0][0]
                similarities.append((other_id, similarity))
        # Sort by similarity and filter by threshold
        similarities = [(aid, sim) for aid, sim in similarities if sim >= similarity_threshold]
        similarities.sort(key=lambda x: x[1], reverse=True)
        # Get the top-k similar agents
        top_similar = similarities[:top_k]
        # Get agent details
        result = []
        for similar_id, similarity in top_similar:
            agent = self.db.query(Agent).filter(Agent.id == similar_id).first()
            if agent:
                result.append({
                    "agent_id": agent.id,
                    "name": agent.name,
                    "similarity_score": float(similarity),
                    "agent_type": agent.agent_type,
                    "capabilities": agent.capabilities,
                })
        return result

    def jaccard_similarity(self, agent_id1: str, agent_id2: str) -> float:
        """Calculate Jaccard similarity between two agents based on capabilities and tags."""
        agent1 = self.db.query(Agent).filter(Agent.id == agent_id1).first()
        agent2 = self.db.query(Agent).filter(Agent.id == agent_id2).first()
        if not agent1 or not agent2:
            return 0.0
        # Get capabilities
        capabilities1 = set(agent1.capabilities or [])
        capabilities2 = set(agent2.capabilities or [])
        # Get tags from agent cards
        card1 = self.db.query(AgentCard).filter(AgentCard.agent_id == agent_id1).first()
        card2 = self.db.query(AgentCard).filter(AgentCard.agent_id == agent_id2).first()
        tags1 = set(card1.tags or []) if card1 else set()
        tags2 = set(card2.tags or []) if card2 else set()
        # Combine capabilities and tags
        features1 = capabilities1.union(tags1)
        features2 = capabilities2.union(tags2)
        # Calculate Jaccard similarity
        if not features1 and not features2:
            return 0.0
        intersection = len(features1.intersection(features2))
        union = len(features1.union(features2))
        return intersection / union if union > 0 else 0.0

    def find_similar_agents_jaccard(self, agent_id: str, top_k: int = 5, similarity_threshold: float = 0.3) -> List[Dict]:
        """Find similar agents using Jaccard similarity."""
        # Get all agents
        agents = self.db.query(Agent).all()
        # Calculate Jaccard similarity with all other agents
        similarities = []
        for agent in agents:
            if agent.id != agent_id:
                similarity = self.jaccard_similarity(agent_id, agent.id)
                if similarity >= similarity_threshold:
                    similarities.append((agent.id, similarity))
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        # Get the top-k similar agents
        top_similar = similarities[:top_k]
        # Get agent details
        result = []
        for similar_id, similarity in top_similar:
            agent = self.db.query(Agent).filter(Agent.id == similar_id).first()
            if agent:
                result.append({
                    "agent_id": agent.id,
                    "name": agent.name,
                    "similarity_score": similarity,
                    "agent_type": agent.agent_type,
                    "capabilities": agent.capabilities,
                })
        return result