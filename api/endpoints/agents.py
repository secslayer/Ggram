from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from api import schemas
from core.database import get_db
from core.security import get_current_active_user
from models.user import User
from services.agent_service import AgentService

router = APIRouter()


@router.post("/", response_model=schemas.Agent, status_code=status.HTTP_201_CREATED)
def create_agent(
    agent_in: schemas.AgentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Create a new agent."""
    # Check if agent with this name already exists for this user
    agent = AgentService.get_by_name(db, name=agent_in.name)
    if agent and agent.owner_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An agent with this name already exists.",
        )
    
    # Create the agent
    agent = AgentService.create(
        db=db,
        user_id=current_user.id,
        name=agent_in.name,
        description=agent_in.description,
        agent_type=agent_in.agent_type,
        llm_config=agent_in.llm_config,
        capabilities=agent_in.capabilities,
        a2a_enabled=agent_in.a2a_enabled,
        a2a_endpoint=agent_in.a2a_endpoint,
    )
    return agent


@router.get("/", response_model=List[schemas.Agent])
def read_agents(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Retrieve agents."""
    # Get all agents for the current user
    agents = AgentService.get_by_user(db, user_id=current_user.id, skip=skip, limit=limit)
    return agents


@router.get("/{agent_id}", response_model=schemas.Agent)
def read_agent(
    agent_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get a specific agent by id."""
    agent = AgentService.get_by_id(db, agent_id=agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found",
        )
    
    # Check if the agent belongs to the current user
    if agent.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    
    return agent


@router.put("/{agent_id}", response_model=schemas.Agent)
def update_agent(
    agent_id: str,
    agent_in: schemas.AgentUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Update an agent."""
    agent = AgentService.get_by_id(db, agent_id=agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found",
        )
    
    # Check if the agent belongs to the current user
    if agent.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    
    # Check if name is being updated and already exists
    if agent_in.name and agent_in.name != agent.name:
        existing_agent = AgentService.get_by_name(db, name=agent_in.name)
        if existing_agent and existing_agent.owner_id == current_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="An agent with this name already exists.",
            )
    
    # Update the agent
    agent = AgentService.update(
        db=db,
        agent_id=agent_id,
        **agent_in.dict(exclude_unset=True),
    )
    return agent


@router.delete("/{agent_id}", response_model=schemas.Agent)
def delete_agent(
    agent_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Delete an agent."""
    agent = AgentService.get_by_id(db, agent_id=agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found",
        )
    
    # Check if the agent belongs to the current user
    if agent.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    
    AgentService.delete(db, agent_id=agent_id)
    return agent


@router.post("/{agent_id}/run", response_model=schemas.AgentRunResponse)
def run_agent(
    agent_id: str,
    run_request: schemas.AgentRunRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Run an agent with the given input."""
    agent = AgentService.get_by_id(db, agent_id=agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found",
        )
    
    # Create an agent instance
    agent_instance = AgentService.create_agent_instance(db, agent_id=agent_id)
    if not agent_instance:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create agent instance",
        )
    
    # Run the agent
    try:
        if not run_request.stream:
            output = agent_instance.run(run_request.input)
            return {"output": output}
        else:
            # Return a streaming response
            return StreamingResponse(
                agent_instance.stream(run_request.input),
                media_type="text/event-stream",
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {str(e)}",
        )