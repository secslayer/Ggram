import uuid
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from core.database import Base


class Agent(Base):
    """Agent model for AI agents created by users."""

    __tablename__ = "agents"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, index=True)
    description = Column(Text)
    owner_id = Column(String, ForeignKey("users.id"))
    is_public = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Agent configuration
    agent_type = Column(String, default="simple")
    llm_model = Column(String)
    llm_temperature = Column(Float, default=0.2)
    llm_top_p = Column(Float, default=0.7)
    llm_max_tokens = Column(Integer, default=1024)
    
    # Agent capabilities and configuration
    capabilities = Column(JSON, default=dict)
    tools = Column(JSON, default=list)
    prompt_template = Column(Text)
    embedding_vector = Column(JSON, nullable=True)  # For similarity search
    
    # A2A protocol configuration
    a2a_enabled = Column(Boolean, default=False)
    a2a_endpoint = Column(String, nullable=True)
    a2a_config = Column(JSON, default=dict)
    
    # Statistics
    usage_count = Column(Integer, default=0)
    avg_rating = Column(Float, default=0.0)
    
    # Relationships
    owner = relationship("User", back_populates="agents")
    card = relationship("AgentCard", back_populates="agent", uselist=False)
    feedbacks = relationship("Feedback", back_populates="agent")
    collaborations_as_initiator = relationship(
        "Collaboration", foreign_keys="[Collaboration.initiator_agent_id]", back_populates="initiator_agent"
    )
    collaborations_as_collaborator = relationship(
        "Collaboration", foreign_keys="[Collaboration.collaborator_agent_id]", back_populates="collaborator_agent"
    )


class AgentCard(Base):
    """Agent card model for displaying agent capabilities and inputs/outputs."""

    __tablename__ = "agent_cards"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    agent_id = Column(String, ForeignKey("agents.id"), unique=True)
    title = Column(String)
    summary = Column(Text)
    input_schema = Column(JSON)  # JSON schema for agent inputs
    output_schema = Column(JSON)  # JSON schema for agent outputs
    example_inputs = Column(JSON, default=list)
    example_outputs = Column(JSON, default=list)
    tags = Column(JSON, default=list)  # List of tags for categorization
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    agent = relationship("Agent", back_populates="card")