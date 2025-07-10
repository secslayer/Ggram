from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api import schemas
from core.database import get_db
from core.security import get_current_active_user
from models.user import User
from services.agent_service import AgentService
from services.collaboration_service import CollaborationService

router = APIRouter()


@router.post("/", response_model=schemas.Collaboration, status_code=status.HTTP_201_CREATED)
def create_collaboration(
    collab_in: schemas.CollaborationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Create a new collaboration."""
    # Check if both agents exist
    initiator_agent = AgentService.get_by_id(db, agent_id=collab_in.initiator_agent_id)
    collaborator_agent = AgentService.get_by_id(db, agent_id=collab_in.collaborator_agent_id)
    
    if not initiator_agent or not collaborator_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="One or both agents not found",
        )
    
    # Check if the initiator agent belongs to the current user
    if initiator_agent.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    
    # Create the collaboration
    collaboration = CollaborationService.create(
        db=db,
        initiator_agent_id=collab_in.initiator_agent_id,
        collaborator_agent_id=collab_in.collaborator_agent_id,
        status=collab_in.status,
    )
    return collaboration


@router.get("/", response_model=List[schemas.Collaboration])
def read_collaborations(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Retrieve collaborations."""
    # Get all collaborations for the current user's agents
    user_agents = AgentService.get_by_user(db, user_id=current_user.id)
    user_agent_ids = [agent.id for agent in user_agents]
    
    collaborations = []
    for agent_id in user_agent_ids:
        agent_collaborations = CollaborationService.get_by_agent(db, agent_id=agent_id, skip=skip, limit=limit)
        collaborations.extend(agent_collaborations)
    
    return collaborations


@router.get("/{collaboration_id}", response_model=schemas.Collaboration)
def read_collaboration(
    collaboration_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get a specific collaboration by id."""
    collaboration = CollaborationService.get_by_id(db, collaboration_id=collaboration_id)
    if not collaboration:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collaboration not found",
        )
    
    # Check if the collaboration involves an agent owned by the current user
    initiator_agent = AgentService.get_by_id(db, agent_id=collaboration.initiator_agent_id)
    collaborator_agent = AgentService.get_by_id(db, agent_id=collaboration.collaborator_agent_id)
    
    if (initiator_agent and initiator_agent.user_id == current_user.id) or \
       (collaborator_agent and collaborator_agent.user_id == current_user.id):
        return collaboration
    
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Not enough permissions",
    )


@router.put("/{collaboration_id}", response_model=schemas.Collaboration)
def update_collaboration(
    collaboration_id: int,
    collab_in: schemas.CollaborationUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Update a collaboration."""
    collaboration = CollaborationService.get_by_id(db, collaboration_id=collaboration_id)
    if not collaboration:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collaboration not found",
        )
    
    # Check if the collaboration involves an agent owned by the current user
    initiator_agent = AgentService.get_by_id(db, agent_id=collaboration.initiator_agent_id)
    
    if not initiator_agent or initiator_agent.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    
    # Update the collaboration
    collaboration = CollaborationService.update(
        db=db,
        collaboration_id=collaboration_id,
        **collab_in.dict(exclude_unset=True),
    )
    return collaboration


@router.delete("/{collaboration_id}", response_model=schemas.Collaboration)
def delete_collaboration(
    collaboration_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Delete a collaboration."""
    collaboration = CollaborationService.get_by_id(db, collaboration_id=collaboration_id)
    if not collaboration:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collaboration not found",
        )
    
    # Check if the collaboration involves an agent owned by the current user
    initiator_agent = AgentService.get_by_id(db, agent_id=collaboration.initiator_agent_id)
    
    if not initiator_agent or initiator_agent.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    
    CollaborationService.delete(db, collaboration_id=collaboration_id)
    return collaboration


@router.post("/{collaboration_id}/execute", response_model=schemas.CollaborationExecuteResponse)
async def execute_collaboration(
    collaboration_id: int,
    execute_request: schemas.CollaborationExecuteRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Execute a collaboration."""
    collaboration = CollaborationService.get_by_id(db, collaboration_id=collaboration_id)
    if not collaboration:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collaboration not found",
        )
    
    # Check if the collaboration involves an agent owned by the current user
    initiator_agent = AgentService.get_by_id(db, agent_id=collaboration.initiator_agent_id)
    
    if not initiator_agent or initiator_agent.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    
    # Check if the collaboration is active
    if collaboration.status != "active":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Collaboration is not active (status: {collaboration.status})",
        )
    
    # Execute the collaboration
    collaboration, result = await CollaborationService.execute_collaboration(
        db=db,
        collaboration_id=collaboration_id,
        input_data=execute_request.input_data,
    )
    
    # Return the result
    return {
        "collaboration_id": collaboration.id,
        "status": collaboration.status,
        "output_data": collaboration.output_data,
        "error_message": collaboration.error_message,
    }


@router.post("/initiate", response_model=schemas.CollaborationExecuteResponse)
async def initiate_collaboration(
    initiator_agent_id: str,
    collaborator_agent_id: str,
    input_data: Dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Initiate a collaboration between two agents."""
    # Check if both agents exist
    initiator_agent = AgentService.get_by_id(db, agent_id=initiator_agent_id)
    collaborator_agent = AgentService.get_by_id(db, agent_id=collaborator_agent_id)
    if not initiator_agent or not collaborator_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="One or both agents not found",
        )
    
    # Check if the initiator agent belongs to the current user
    if initiator_agent.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    
    # Initiate the collaboration
    collaboration, result = await CollaborationService.initiate_collaboration(
        db=db,
        initiator_agent_id=initiator_agent_id,
        collaborator_agent_id=collaborator_agent_id,
        input_data=input_data,
    )
    
    # Return the result
    return {
        "collaboration_id": collaboration.id,
        "status": collaboration.status,
        "output_data": result if isinstance(result, dict) else None,
        "error_message": result if isinstance(result, str) else None,
    }