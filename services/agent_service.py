from typing import Dict, List, Optional, Union

from sqlalchemy.orm import Session

from agents.factory import AgentFactory
from core.config import settings
from models.agent import Agent, AgentCard
from models.user import User


class AgentService:
    """Service for agent management."""

    @staticmethod
    def get_by_id(db: Session, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID."""
        return db.query(Agent).filter(Agent.id == agent_id).first()

    @staticmethod
    def get_by_name(db: Session, name: str) -> Optional[Agent]:
        """Get an agent by name."""
        return db.query(Agent).filter(Agent.name == name).first()

    @staticmethod
    def get_all(db: Session, skip: int = 0, limit: int = 100) -> List[Agent]:
        """Get all agents with pagination."""
        return db.query(Agent).offset(skip).limit(limit).all()

    @staticmethod
    def get_by_user(db: Session, user_id: str, skip: int = 0, limit: int = 100) -> List[Agent]:
        """Get all agents for a user."""
        return db.query(Agent).filter(Agent.owner_id == user_id).offset(skip).limit(limit).all()

    @staticmethod
    def create(
        db: Session,
        user_id: str,
        name: str,
        description: str,
        agent_type: str,
        llm_config: Dict,
        capabilities: List[str],
        a2a_enabled: bool = False,
        a2a_endpoint: Optional[str] = None,
        **kwargs,
    ) -> Agent:
        """Create a new agent."""
        # Create the agent in the database
        db_agent = Agent(
            owner_id=user_id,
            name=name,
            description=description,
            agent_type=agent_type,
            llm_model=llm_config.get("model"),
            llm_temperature=llm_config.get("temperature", 0.2),
            llm_top_p=llm_config.get("top_p", 0.7),
            llm_max_tokens=llm_config.get("max_tokens", 1024),
            capabilities=capabilities,
            a2a_enabled=a2a_enabled,
            a2a_endpoint=a2a_endpoint,
            a2a_config={"enabled": a2a_enabled} if a2a_enabled else {},
        )
        db.add(db_agent)
        db.commit()
        db.refresh(db_agent)
        return db_agent

    @staticmethod
    def update(db: Session, agent_id: str, **kwargs) -> Optional[Agent]:
        """Update an agent."""
        db_agent = AgentService.get_by_id(db, agent_id)
        if not db_agent:
            return None

        # Update agent attributes
        for key, value in kwargs.items():
            if hasattr(db_agent, key):
                setattr(db_agent, key, value)

        db.commit()
        db.refresh(db_agent)
        return db_agent

    @staticmethod
    def delete(db: Session, agent_id: str) -> bool:
        """Delete an agent."""
        db_agent = AgentService.get_by_id(db, agent_id)
        if not db_agent:
            return False

        db.delete(db_agent)
        db.commit()
        return True

    @staticmethod
    def create_agent_instance(db: Session, agent_id: str) -> Optional[Union[Agent, object]]:
        """Create a runtime agent instance from a database agent."""
        db_agent = AgentService.get_by_id(db, agent_id)
        if not db_agent:
            return None

        # Create the agent instance using the factory
        try:
            # Extract LLM configuration
            llm_config = getattr(db_agent, "llm_config", None) or {}
            if not llm_config:
                llm_config = {
                    "model": db_agent.llm_model,
                    "temperature": db_agent.llm_temperature,
                    "top_p": db_agent.llm_top_p,
                    "max_tokens": db_agent.llm_max_tokens,
                }
            llm_config.setdefault("api_key", settings.nvidia_api_key)
            llm_config.setdefault("model", settings.DEFAULT_LLM_MODEL)
            # Create the agent instance
            agent_instance = AgentFactory.create_agent(
                agent_type=db_agent.agent_type,
                name=db_agent.name,
                description=db_agent.description,
                **llm_config,
            )
            return agent_instance
        except Exception as e:
            # Log the error
            print(f"Error creating agent instance: {e}")
            return None



async def get_agent_instance(agent_id: str, db: Session):
    """Get an agent instance by ID."""
    return AgentService.create_agent_instance(db, agent_id)


class AgentCardService:
    """Service for agent card management."""

    @staticmethod
    def get_by_id(db: Session, card_id: int) -> Optional[AgentCard]:
        """Get an agent card by ID."""
        return db.query(AgentCard).filter(AgentCard.id == card_id).first()

    @staticmethod
    def get_by_agent(db: Session, agent_id: int) -> Optional[AgentCard]:
        """Get an agent card by agent ID."""
        return db.query(AgentCard).filter(AgentCard.agent_id == agent_id).first()

    @staticmethod
    def get_all(db: Session, skip: int = 0, limit: int = 100) -> List[AgentCard]:
        """Get all agent cards with pagination."""
        return db.query(AgentCard).offset(skip).limit(limit).all()

    @staticmethod
    def create(
        db: Session,
        agent_id: int,
        title: str,
        summary: str,
        input_schema: Optional[Dict] = None,
        output_schema: Optional[Dict] = None,
        examples: Optional[List[Dict]] = None,
        tags: Optional[List[str]] = None,
        image_url: Optional[str] = None,
    ) -> AgentCard:
        """Create a new agent card."""
        db_card = AgentCard(
            agent_id=agent_id,
            title=title,
            summary=summary,
            input_schema=input_schema,
            output_schema=output_schema,
            examples=examples,
            tags=tags,
            image_url=image_url,
        )
        db.add(db_card)
        db.commit()
        db.refresh(db_card)
        return db_card

    @staticmethod
    def update(db: Session, card_id: int, **kwargs) -> Optional[AgentCard]:
        """Update an agent card."""
        db_card = AgentCardService.get_by_id(db, card_id)
        if not db_card:
            return None

        # Update card attributes
        for key, value in kwargs.items():
            if hasattr(db_card, key):
                setattr(db_card, key, value)

        db.commit()
        db.refresh(db_card)
        return db_card

    @staticmethod
    def delete(db: Session, card_id: int) -> bool:
        """Delete an agent card."""
        db_card = AgentCardService.get_by_id(db, card_id)
        if not db_card:
            return False

        db.delete(db_card)
        db.commit()
        return True