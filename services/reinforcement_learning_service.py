import numpy as np
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import torch
import torch.nn.functional as F
from sqlalchemy.orm import Session
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from models.agent import Agent
from models.collaboration import Collaboration
from models.feedback import Feedback


class GNNReinforcementLearningModel(torch.nn.Module):
    """Graph Neural Network model for reinforcement learning."""

    def __init__(self, num_features: int, hidden_channels: int, num_actions: int):
        super(GNNReinforcementLearningModel, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, num_actions)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x


class ReinforcementLearningService:
    """Service for graph reinforcement learning."""

    def __init__(self, db: Session):
        self.db = db
        self.model = None
        self.optimizer = None
        self.initialized = False
        self.agent_ids = None
        self.action_space = None

    def _initialize_model(self, num_features: int, num_actions: int):
        """Initialize the GNN model for reinforcement learning."""
        hidden_channels = 64
        self.model = GNNReinforcementLearningModel(num_features, hidden_channels, num_actions)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def _create_agent_features(self, agents: List[Agent]) -> Tuple[torch.Tensor, List[int]]:
        """Create feature vectors for agents."""
        features = []
        agent_ids = []

        # Get all unique capabilities
        all_capabilities = set()
        for agent in agents:
            if agent.capabilities:
                all_capabilities.update(agent.capabilities)

        # Create a mapping from capability to index
        capability_to_idx = {cap: i for i, cap in enumerate(all_capabilities)}
        
        # Feature vector size: capabilities + agent_type one-hot + average rating
        agent_types = set(agent.agent_type for agent in agents)
        agent_type_to_idx = {t: i for i, t in enumerate(agent_types)}
        feature_size = len(capability_to_idx) + len(agent_type_to_idx) + 1  # +1 for rating

        # Create feature vectors
        for agent in agents:
            feature = np.zeros(feature_size)
            
            # Set capability features
            if agent.capabilities:
                for cap in agent.capabilities:
                    if cap in capability_to_idx:
                        feature[capability_to_idx[cap]] = 1
            
            # Set agent type one-hot
            type_idx = len(capability_to_idx) + agent_type_to_idx[agent.agent_type]
            feature[type_idx] = 1
            
            # Set average rating
            avg_rating = self.db.query(Feedback).filter(Feedback.agent_id == agent.id).with_entities(
                Feedback.rating
            ).all()
            if avg_rating:
                feature[-1] = sum(r[0] for r in avg_rating) / len(avg_rating) / 5.0  # Normalize to [0, 1]
            
            features.append(feature)
            agent_ids.append(agent.id)

        return torch.tensor(features, dtype=torch.float), agent_ids

    def _create_agent_graph(self, agents: List[Agent]) -> Tuple[nx.Graph, List[Tuple[int, int]]]:
        """Create a graph of agents based on collaborations."""
        graph = nx.Graph()
        action_space = []  # Pairs of agent indices that can collaborate
        
        # Add nodes
        for i, agent in enumerate(agents):
            graph.add_node(i, agent_id=agent.id)
        
        # Add edges based on existing collaborations
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents):
                if i != j:
                    # Check if there's a collaboration between these agents
                    collaboration = self.db.query(Collaboration).filter(
                        ((Collaboration.initiator_agent_id == agent1.id) & 
                         (Collaboration.collaborator_agent_id == agent2.id)) |
                        ((Collaboration.initiator_agent_id == agent2.id) & 
                         (Collaboration.collaborator_agent_id == agent1.id))
                    ).first()
                    
                    if collaboration and collaboration.status in ["active", "completed"]:
                        graph.add_edge(i, j)
                    
                    # Add to action space if both agents support A2A
                    if agent1.a2a_enabled and agent2.a2a_enabled:
                        action_space.append((i, j))
        
        return graph, action_space

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

    def _calculate_rewards(self, agents: List[Agent]) -> torch.Tensor:
        """Calculate rewards for each agent based on feedback and collaboration success."""
        rewards = torch.zeros(len(agents))
        
        for i, agent in enumerate(agents):
            # Get average rating from feedback
            avg_rating = self.db.query(Feedback).filter(Feedback.agent_id == agent.id).with_entities(
                Feedback.rating
            ).all()
            if avg_rating:
                rewards[i] += sum(r[0] for r in avg_rating) / len(avg_rating) / 5.0  # Normalize to [0, 1]
            
            # Get collaboration success rate
            collaborations = self.db.query(Collaboration).filter(
                (Collaboration.initiator_agent_id == agent.id) | 
                (Collaboration.collaborator_agent_id == agent.id)
            ).all()
            
            if collaborations:
                success_count = sum(1 for c in collaborations if c.status == "completed")
                rewards[i] += success_count / len(collaborations)
        
        return rewards

    def build_reinforcement_learning_model(self):
        """Build the reinforcement learning model from the current agents."""
        # Get all agents
        agents = self.db.query(Agent).all()
        if not agents:
            self.initialized = False
            return
        
        # Create feature vectors
        features, agent_ids = self._create_agent_features(agents)
        
        # Create agent graph and action space
        graph, action_space = self._create_agent_graph(agents)
        
        # Convert to PyTorch Geometric format
        data = self._convert_to_pytorch_geometric(graph, features)
        
        # Initialize model if needed
        if self.model is None or features.shape[1] != self.model.conv1.in_channels or len(action_space) != self.model.conv3.out_channels:
            self._initialize_model(features.shape[1], len(action_space))
        
        self.agent_ids = agent_ids
        self.action_space = action_space
        self.initialized = True
        
        return data

    def train(self, num_epochs: int = 100):
        """Train the reinforcement learning model."""
        if not self.initialized:
            data = self.build_reinforcement_learning_model()
            if not self.initialized:
                return
            # Get all agents
            agents = self.db.query(Agent).filter(Agent.id.in_(self.agent_ids)).all()
            # Create feature vectors
            features, _ = self._create_agent_features(agents)
            # Create agent graph
            graph, _ = self._create_agent_graph(agents)
            # Convert to PyTorch Geometric format
            data = self._convert_to_pytorch_geometric(graph, features)
        else:
            # Get all agents
            agents = self.db.query(Agent).filter(Agent.id.in_(self.agent_ids)).all()
            # Create feature vectors
            features, _ = self._create_agent_features(agents)
            # Create agent graph
            graph, _ = self._create_agent_graph(agents)
            # Convert to PyTorch Geometric format
            data = self._convert_to_pytorch_geometric(graph, features)
        
        # Calculate rewards
        rewards = self._calculate_rewards(agents)
        
        # Train the model
        self.model.train()
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)
            
            # Calculate loss (policy gradient)
            action_probs = F.softmax(out, dim=1)
            log_probs = F.log_softmax(out, dim=1)
            loss = -torch.mean(rewards * log_probs.sum(dim=1))
            
            loss.backward()
            self.optimizer.step()

    def predict_collaboration_success(self, agent_id1: str, agent_id2: str) -> float:
        """Predict the success probability of a collaboration between two agents."""
        if not self.initialized:
            self.build_reinforcement_learning_model()
            if not self.initialized:
                return 0.0
        # Check if both agents are in the model
        if agent_id1 not in self.agent_ids or agent_id2 not in self.agent_ids:
            return 0.0
        # Get agent indices
        idx1 = self.agent_ids.index(agent_id1)
        idx2 = self.agent_ids.index(agent_id2)
        # Check if this collaboration is in the action space
        action_idx = None
        for i, (a1, a2) in enumerate(self.action_space):
            if (a1 == idx1 and a2 == idx2) or (a1 == idx2 and a2 == idx1):
                action_idx = i
                break
        if action_idx is None:
            return 0.0
        # Get all agents
        agents = self.db.query(Agent).filter(Agent.id.in_(self.agent_ids)).all()
        # Create feature vectors
        features, _ = self._create_agent_features(agents)
        # Create agent graph
        graph, _ = self._create_agent_graph(agents)
        # Convert to PyTorch Geometric format
        data = self._convert_to_pytorch_geometric(graph, features)
        # Predict
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            action_probs = F.softmax(out, dim=1)
            # Get the probability for this collaboration
            prob = action_probs[idx1, action_idx].item()
            return prob

    def recommend_collaborations(self, agent_id: str, top_k: int = 5) -> List[Dict]:
        """Recommend potential collaborations for an agent."""
        if not self.initialized:
            self.build_reinforcement_learning_model()
            if not self.initialized:
                return []
        # Check if the agent is in the model
        if agent_id not in self.agent_ids:
            return []
        # Get agent index
        agent_idx = self.agent_ids.index(agent_id)
        # Get all agents
        agents = self.db.query(Agent).filter(Agent.id.in_(self.agent_ids)).all()
        # Create feature vectors
        features, _ = self._create_agent_features(agents)
        # Create agent graph
        graph, _ = self._create_agent_graph(agents)
        # Convert to PyTorch Geometric format
        data = self._convert_to_pytorch_geometric(graph, features)
        # Predict
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            action_probs = F.softmax(out, dim=1)
        # Get probabilities for all possible collaborations with this agent
        collaborations = []
        for i, (a1, a2) in enumerate(self.action_space):
            if a1 == agent_idx or a2 == agent_idx:
                other_idx = a2 if a1 == agent_idx else a1
                other_agent_id = self.agent_ids[other_idx]
                # Check if there's already a collaboration
                existing_collab = self.db.query(Collaboration).filter(
                    ((Collaboration.initiator_agent_id == agent_id) & 
                     (Collaboration.collaborator_agent_id == other_agent_id)) |
                    ((Collaboration.initiator_agent_id == other_agent_id) & 
                     (Collaboration.collaborator_agent_id == agent_id))
                ).first()
                if not existing_collab or existing_collab.status not in ["active", "completed"]:
                    prob = action_probs[agent_idx, i].item()
                    collaborations.append((other_agent_id, prob))
        # Sort by probability
        collaborations.sort(key=lambda x: x[1], reverse=True)
        # Get the top-k recommendations
        top_recommendations = collaborations[:top_k]
        # Get agent details
        result = []
        for other_id, prob in top_recommendations:
            agent = self.db.query(Agent).filter(Agent.id == other_id).first()
            if agent:
                result.append({
                    "agent_id": agent.id,
                    "name": agent.name,
                    "success_probability": prob,
                    "agent_type": agent.agent_type,
                    "capabilities": agent.capabilities,
                })
        return result