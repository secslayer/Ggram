from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api import schemas
from core.database import get_db
from core.security import get_current_active_user
from models.user import User
from services.agent_service import AgentService
from services.reinforcement_learning_service import ReinforcementLearningService
from services.similarity_service import SimilarityService

router = APIRouter()


@router.post("/search", response_model=schemas.SimilaritySearchResponse)
def search_similar_agents(
    search_request: schemas.SimilaritySearchRequest,
    db: Session = Depends(get_db),
) -> Any:
    """Search for similar agents."""
    # Check if the agent exists
    agent = AgentService.get_by_id(db, agent_id=search_request.agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found",
        )
    
    # Create similarity service
    similarity_service = SimilarityService(db)
    
    # Find similar agents
    if search_request.method == "gnn":
        similar_agents = similarity_service.find_similar_agents(
            agent_id=search_request.agent_id,
            top_k=search_request.top_k,
            similarity_threshold=search_request.similarity_threshold,
        )
    elif search_request.method == "jaccard":
        similar_agents = similarity_service.find_similar_agents_jaccard(
            agent_id=search_request.agent_id,
            top_k=search_request.top_k,
            similarity_threshold=search_request.similarity_threshold,
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid similarity method: {search_request.method}",
        )
    
    return {"similar_agents": similar_agents}


@router.post("/recommend", response_model=schemas.CollaborationRecommendationResponse)
def recommend_collaborations(
    recommendation_request: schemas.CollaborationRecommendationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Recommend potential collaborations for an agent."""
    # Check if the agent exists and belongs to the current user
    agent = AgentService.get_by_id(db, agent_id=recommendation_request.agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found",
        )
    
    if agent.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    
    # Create reinforcement learning service
    rl_service = ReinforcementLearningService(db)
    
    # Get recommendations
    recommended_agents = rl_service.recommend_collaborations(
        agent_id=recommendation_request.agent_id,
        top_k=recommendation_request.top_k,
    )
    
    return {"recommended_agents": recommended_agents}


@router.post("/train")
def train_reinforcement_learning(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    num_epochs: int = 100,
) -> Any:
    """Train the reinforcement learning model."""
    # Create reinforcement learning service
    rl_service = ReinforcementLearningService(db)
    
    # Train the model
    rl_service.train(num_epochs=num_epochs)
    
    return {"status": "success", "message": f"Trained for {num_epochs} epochs"}