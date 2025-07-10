from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api import schemas
from core.database import get_db
from core.security import get_current_active_user
from models.user import User
from services.agent_service import AgentCardService, AgentService

router = APIRouter()


@router.post("/", response_model=schemas.AgentCard, status_code=status.HTTP_201_CREATED)
def create_agent_card(
    card_in: schemas.AgentCardCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Create a new agent card."""
    # Check if the agent exists and belongs to the current user
    agent = AgentService.get_by_id(db, agent_id=card_in.agent_id)
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
    
    # Check if a card already exists for this agent
    existing_card = AgentCardService.get_by_agent(db, agent_id=card_in.agent_id)
    if existing_card:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An agent card already exists for this agent",
        )
    
    # Create the agent card
    card = AgentCardService.create(
        db=db,
        agent_id=card_in.agent_id,
        title=card_in.title,
        summary=card_in.summary,
        input_schema=card_in.input_schema,
        output_schema=card_in.output_schema,
        examples=card_in.examples,
        tags=card_in.tags,
        image_url=card_in.image_url,
    )
    return card


@router.get("/", response_model=List[schemas.AgentCard])
def read_agent_cards(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """Retrieve agent cards."""
    cards = AgentCardService.get_all(db, skip=skip, limit=limit)
    return cards


@router.get("/{card_id}", response_model=schemas.AgentCard)
def read_agent_card(
    card_id: int,
    db: Session = Depends(get_db),
) -> Any:
    """Get a specific agent card by id."""
    card = AgentCardService.get_by_id(db, card_id=card_id)
    if not card:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent card not found",
        )
    return card


@router.get("/by-agent/{agent_id}", response_model=schemas.AgentCard)
def read_agent_card_by_agent(
    agent_id: str,
    db: Session = Depends(get_db),
) -> Any:
    """Get a specific agent card by agent id."""
    card = AgentCardService.get_by_agent(db, agent_id=agent_id)
    if not card:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent card not found",
        )
    return card


@router.put("/{card_id}", response_model=schemas.AgentCard)
def update_agent_card(
    card_id: int,
    card_in: schemas.AgentCardUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Update an agent card."""
    card = AgentCardService.get_by_id(db, card_id=card_id)
    if not card:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent card not found",
        )
    
    # Check if the agent belongs to the current user
    agent = AgentService.get_by_id(db, agent_id=card.agent_id)
    if not agent or agent.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    
    # Update the agent card
    card = AgentCardService.update(
        db=db,
        card_id=card_id,
        **card_in.dict(exclude_unset=True),
    )
    return card


@router.delete("/{card_id}", response_model=schemas.AgentCard)
def delete_agent_card(
    card_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Delete an agent card."""
    card = AgentCardService.get_by_id(db, card_id=card_id)
    if not card:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent card not found",
        )
    
    # Check if the agent belongs to the current user
    agent = AgentService.get_by_id(db, agent_id=card.agent_id)
    if not agent or agent.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    
    AgentCardService.delete(db, card_id=card_id)
    return card