from typing import Dict, List, Optional, Tuple, Union

from sqlalchemy.orm import Session

from agents.a2a_client import A2AAgentClient
from models.collaboration import Collaboration
from services.agent_service import AgentService


class CollaborationService:
    """Service for collaboration management."""

    @staticmethod
    def get_by_id(db: Session, collaboration_id: int) -> Optional[Collaboration]:
        """Get a collaboration by ID."""
        return db.query(Collaboration).filter(Collaboration.id == collaboration_id).first()

    @staticmethod
    def get_by_agent(db: Session, agent_id: str, skip: int = 0, limit: int = 100) -> List[Collaboration]:
        """Get all collaborations for an agent."""
        return (
            db.query(Collaboration)
            .filter(
                (Collaboration.initiator_agent_id == agent_id) | 
                (Collaboration.collaborator_agent_id == agent_id)
            )
            .offset(skip)
            .limit(limit)
            .all()
        )

    @staticmethod
    def get_all(db: Session, skip: int = 0, limit: int = 100) -> List[Collaboration]:
        """Get all collaborations with pagination."""
        return db.query(Collaboration).offset(skip).limit(limit).all()

    @staticmethod
    def create(
        db: Session,
        initiator_agent_id: str,
        collaborator_agent_id: str,
        status: str = "pending",
    ) -> Collaboration:
        """Create a new collaboration."""
        db_collaboration = Collaboration(
            initiator_agent_id=initiator_agent_id,
            collaborator_agent_id=collaborator_agent_id,
            status=status,
        )
        db.add(db_collaboration)
        db.commit()
        db.refresh(db_collaboration)
        return db_collaboration

    @staticmethod
    def update(db: Session, collaboration_id: int, **kwargs) -> Optional[Collaboration]:
        """Update a collaboration."""
        db_collaboration = CollaborationService.get_by_id(db, collaboration_id)
        if not db_collaboration:
            return None

        # Update collaboration attributes
        for key, value in kwargs.items():
            if hasattr(db_collaboration, key):
                setattr(db_collaboration, key, value)

        db.commit()
        db.refresh(db_collaboration)
        return db_collaboration

    @staticmethod
    def delete(db: Session, collaboration_id: int) -> bool:
        """Delete a collaboration."""
        db_collaboration = CollaborationService.get_by_id(db, collaboration_id)
        if not db_collaboration:
            return False

        db.delete(db_collaboration)
        db.commit()
        return True

    @staticmethod
    async def initiate_collaboration(
        db: Session,
        initiator_agent_id: str,
        collaborator_agent_id: str,
        input_data: Dict,
    ) -> Tuple[Collaboration, Union[str, Dict]]:
        """Initiate a collaboration between two agents."""
        # Create the collaboration record
        db_collaboration = CollaborationService.create(
            db=db,
            initiator_agent_id=initiator_agent_id,
            collaborator_agent_id=collaborator_agent_id,
            status="negotiating",
        )
        # Get the agents
        initiator_agent = AgentService.get_by_id(db, initiator_agent_id)
        collaborator_agent = AgentService.get_by_id(db, collaborator_agent_id)
        if not initiator_agent or not collaborator_agent:
            CollaborationService.update(db, db_collaboration.id, status="failed", error_message="Agent not found")
            return db_collaboration, "Agent not found"
        
        # Check if both agents support A2A
        if not (initiator_agent.a2a_enabled and collaborator_agent.a2a_enabled):
            CollaborationService.update(
                db, 
                db_collaboration.id, 
                status="failed", 
                error_message="One or both agents do not support A2A"
            )
            return db_collaboration, "One or both agents do not support A2A"
        
        # Create A2A client
        try:
            a2a_client = A2AAgentClient(
                sender_id=str(initiator_agent.id),
                receiver_id=str(collaborator_agent.id),
                receiver_endpoint=collaborator_agent.a2a_endpoint,
            )
            
            # Store the input data
            CollaborationService.update(db, db_collaboration.id, input_data=input_data)
            
            # Send the collaboration request
            negotiation_message = {
                "type": "collaboration_request",
                "initiator_id": initiator_agent.id,
                "initiator_name": initiator_agent.name,
                "collaboration_id": db_collaboration.id,
                "input_data": input_data,
            }
            
            # Run the task
            result = await a2a_client.run_task(input_text=str(negotiation_message))
            
            # Update the collaboration with the negotiation history
            negotiation_history = [
                {"sender": "initiator", "message": negotiation_message},
                {"sender": "collaborator", "message": result},
            ]
            
            # Check if the collaboration was accepted
            if isinstance(result, dict) and result.get("accepted", False):
                CollaborationService.update(
                    db, 
                    db_collaboration.id, 
                    status="active",
                    negotiation_history=negotiation_history,
                )
                return db_collaboration, result
            else:
                CollaborationService.update(
                    db, 
                    db_collaboration.id, 
                    status="rejected",
                    negotiation_history=negotiation_history,
                )
                return db_collaboration, result
            
        except Exception as e:
            # Update the collaboration with the error
            CollaborationService.update(
                db, 
                db_collaboration.id, 
                status="failed",
                error_message=str(e),
            )
            return db_collaboration, str(e)

    @staticmethod
    async def execute_collaboration(
        db: Session,
        collaboration_id: int,
        input_data: Dict,
    ) -> Tuple[Collaboration, Union[str, Dict]]:
        """Execute a collaboration between two agents."""
        # Get the collaboration
        db_collaboration = CollaborationService.get_by_id(db, collaboration_id)
        if not db_collaboration:
            return None, "Collaboration not found"
        
        # Check if the collaboration is active
        if db_collaboration.status != "active":
            return db_collaboration, f"Collaboration is not active (status: {db_collaboration.status})"
        
        # Get the agents
        initiator_agent = AgentService.get_by_id(db, db_collaboration.initiator_agent_id)
        collaborator_agent = AgentService.get_by_id(db, db_collaboration.collaborator_agent_id)
        
        if not initiator_agent or not collaborator_agent:
            CollaborationService.update(db, db_collaboration.id, status="failed", error_message="Agent not found")
            return db_collaboration, "Agent not found"
        
        # Create A2A client
        try:
            a2a_client = A2AAgentClient(
                sender_id=str(initiator_agent.id),
                receiver_id=str(collaborator_agent.id),
                receiver_endpoint=collaborator_agent.a2a_endpoint,
            )
            
            # Store the input data
            CollaborationService.update(
                db, 
                db_collaboration.id, 
                input_data=input_data,
                status="executing",
            )
            
            # Run the task
            result = await a2a_client.run_task(input_text=str(input_data))
            
            # Update the collaboration with the result
            CollaborationService.update(
                db, 
                db_collaboration.id, 
                status="completed",
                output_data=result,
            )
            
            return db_collaboration, result
            
        except Exception as e:
            # Update the collaboration with the error
            CollaborationService.update(
                db, 
                db_collaboration.id, 
                status="failed",
                error_message=str(e),
            )
            return db_collaboration, str(e)