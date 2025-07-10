from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api import schemas
from core.database import get_db
from core.security import get_current_active_user
from models.user import User
from services.feedback_service import FeedbackService

router = APIRouter()


@router.post("/", response_model=schemas.Feedback, status_code=status.HTTP_201_CREATED)
def create_feedback(
    feedback_in: schemas.FeedbackCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Create new feedback."""
    # Create the feedback
    feedback = FeedbackService.create(
        db=db,
        user_id=current_user.id,
        agent_id=feedback_in.agent_id,
        rating=feedback_in.rating,
        comments=feedback_in.comments,
        collaboration_id=feedback_in.collaboration_id,
    )
    return feedback


@router.get("/", response_model=List[schemas.Feedback])
def read_feedback(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Retrieve feedback."""
    # Get all feedback from the current user
    feedback = FeedbackService.get_by_user(db, user_id=current_user.id, skip=skip, limit=limit)
    return feedback


@router.get("/by-agent/{agent_id}", response_model=List[schemas.Feedback])
def read_feedback_by_agent(
    agent_id: str,
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """Retrieve feedback for a specific agent."""
    feedback = FeedbackService.get_by_agent(db, agent_id=agent_id, skip=skip, limit=limit)
    return feedback


@router.get("/by-collaboration/{collaboration_id}", response_model=List[schemas.Feedback])
def read_feedback_by_collaboration(
    collaboration_id: int,
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """Retrieve feedback for a specific collaboration."""
    feedback = FeedbackService.get_by_collaboration(db, collaboration_id=collaboration_id, skip=skip, limit=limit)
    return feedback


@router.get("/{feedback_id}", response_model=schemas.Feedback)
def read_feedback_by_id(
    feedback_id: int,
    db: Session = Depends(get_db),
) -> Any:
    """Get a specific feedback by id."""
    feedback = FeedbackService.get_by_id(db, feedback_id=feedback_id)
    if not feedback:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feedback not found",
        )
    return feedback


@router.put("/{feedback_id}", response_model=schemas.Feedback)
def update_feedback(
    feedback_id: int,
    feedback_in: schemas.FeedbackUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Update feedback."""
    feedback = FeedbackService.get_by_id(db, feedback_id=feedback_id)
    if not feedback:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feedback not found",
        )
    
    # Check if the feedback belongs to the current user
    if feedback.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    
    # Update the feedback
    feedback = FeedbackService.update(
        db=db,
        feedback_id=feedback_id,
        **feedback_in.dict(exclude_unset=True),
    )
    return feedback


@router.delete("/{feedback_id}", response_model=schemas.Feedback)
def delete_feedback(
    feedback_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Delete feedback."""
    feedback = FeedbackService.get_by_id(db, feedback_id=feedback_id)
    if not feedback:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feedback not found",
        )
    
    # Check if the feedback belongs to the current user
    if feedback.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    
    FeedbackService.delete(db, feedback_id=feedback_id)
    return feedback