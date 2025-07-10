import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from core.database import Base


class Feedback(Base):
    """Feedback model for user feedback on agents."""

    __tablename__ = "feedbacks"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"))
    agent_id = Column(String, ForeignKey("agents.id"))
    collaboration_id = Column(String, ForeignKey("collaborations.id"), nullable=True)
    rating = Column(Integer)  # 1-5 star rating
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="feedbacks")
    agent = relationship("Agent", back_populates="feedbacks")
    collaboration = relationship("Collaboration", back_populates="feedbacks")