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


class Collaboration(Base):
    """Collaboration model for agent collaborations."""

    __tablename__ = "collaborations"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    initiator_agent_id = Column(String, ForeignKey("agents.id"))
    collaborator_agent_id = Column(String, ForeignKey("agents.id"))
    status = Column(String)  # pending, active, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Collaboration details
    input_data = Column(JSON, default=dict)
    output_data = Column(JSON, nullable=True)
    negotiation_history = Column(JSON, default=list)  # List of negotiation messages
    error_message = Column(Text, nullable=True)
    
    # Statistics
    execution_time_ms = Column(Integer, nullable=True)
    avg_rating = Column(Float, default=0.0)
    
    # Relationships
    initiator_agent = relationship(
        "Agent", foreign_keys=[initiator_agent_id], back_populates="collaborations_as_initiator"
    )
    collaborator_agent = relationship(
        "Agent", foreign_keys=[collaborator_agent_id], back_populates="collaborations_as_collaborator"
    )
    feedbacks = relationship("Feedback", back_populates="collaboration")