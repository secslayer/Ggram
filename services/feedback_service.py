from typing import List, Optional

from sqlalchemy.orm import Session

from models.feedback import Feedback


class FeedbackService:
    """Service for feedback management."""

    @staticmethod
    def get_by_id(db: Session, feedback_id: int) -> Optional[Feedback]:
        """Get feedback by ID."""
        return db.query(Feedback).filter(Feedback.id == feedback_id).first()

    @staticmethod
    def get_by_user(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[Feedback]:
        """Get all feedback from a user."""
        return db.query(Feedback).filter(Feedback.user_id == user_id).offset(skip).limit(limit).all()

    @staticmethod
    def get_by_agent(db: Session, agent_id: str, skip: int = 0, limit: int = 100) -> List[Feedback]:
        """Get all feedback for an agent."""
        return db.query(Feedback).filter(Feedback.agent_id == agent_id).offset(skip).limit(limit).all()

    @staticmethod
    def get_by_collaboration(db: Session, collaboration_id: int, skip: int = 0, limit: int = 100) -> List[Feedback]:
        """Get all feedback for a collaboration."""
        return db.query(Feedback).filter(Feedback.collaboration_id == collaboration_id).offset(skip).limit(limit).all()

    @staticmethod
    def create(
        db: Session,
        user_id: int,
        agent_id: str,
        rating: int,
        comments: Optional[str] = None,
        collaboration_id: Optional[int] = None,
    ) -> Feedback:
        """Create new feedback."""
        db_feedback = Feedback(
            user_id=user_id,
            agent_id=agent_id,
            rating=rating,
            comments=comments,
            collaboration_id=collaboration_id,
        )
        db.add(db_feedback)
        db.commit()
        db.refresh(db_feedback)
        return db_feedback

    @staticmethod
    def update(db: Session, feedback_id: int, **kwargs) -> Optional[Feedback]:
        """Update feedback."""
        db_feedback = FeedbackService.get_by_id(db, feedback_id)
        if not db_feedback:
            return None

        # Update feedback attributes
        for key, value in kwargs.items():
            if hasattr(db_feedback, key):
                setattr(db_feedback, key, value)

        db.commit()
        db.refresh(db_feedback)
        return db_feedback

    @staticmethod
    def delete(db: Session, feedback_id: int) -> bool:
        """Delete feedback."""
        db_feedback = FeedbackService.get_by_id(db, feedback_id)
        if not db_feedback:
            return False

        db.delete(db_feedback)
        db.commit()
        return True

    @staticmethod
    def get_average_rating(db: Session, agent_id: str) -> float:
        """Get the average rating for an agent."""
        from sqlalchemy import func
        result = db.query(func.avg(Feedback.rating)).filter(Feedback.agent_id == agent_id).scalar()
        return float(result) if result is not None else 0.0

    @staticmethod
    def get_feedback_count(db: Session, agent_id: str) -> int:
        """Get the number of feedback entries for an agent."""
        return db.query(Feedback).filter(Feedback.agent_id == agent_id).count()