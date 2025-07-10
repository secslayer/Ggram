from datetime import datetime
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, EmailStr, Field, validator


# User schemas
class UserBase(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None


class UserCreate(UserBase):
    password: str


class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = None


class UserInDB(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class User(UserInDB):
    pass


# Token schemas
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenPayload(BaseModel):
    sub: Optional[int] = None


# Agent schemas
class AgentBase(BaseModel):
    name: str
    description: str
    agent_type: str
    capabilities: List[str] = []
    a2a_enabled: bool = False
    a2a_endpoint: Optional[str] = None


class AgentCreate(AgentBase):
    llm_config: Dict = Field(default_factory=dict)


class AgentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    agent_type: Optional[str] = None
    capabilities: Optional[List[str]] = None
    llm_config: Optional[Dict] = None
    a2a_enabled: Optional[bool] = None
    a2a_endpoint: Optional[str] = None


class AgentInDB(AgentBase):
    id: str
    owner_id: str
    agent_type: str
    llm_model: str
    llm_temperature: float
    llm_top_p: float
    llm_max_tokens: int
    tools: List = []
    a2a_config: Dict = {}
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class Agent(AgentInDB):
    pass


# Agent Card schemas
class AgentCardBase(BaseModel):
    title: str
    summary: str
    input_schema: Optional[Dict] = None
    output_schema: Optional[Dict] = None
    examples: Optional[List[Dict]] = None
    tags: Optional[List[str]] = None
    image_url: Optional[str] = None


class AgentCardCreate(AgentCardBase):
    agent_id: str


class AgentCardUpdate(BaseModel):
    title: Optional[str] = None
    summary: Optional[str] = None
    input_schema: Optional[Dict] = None
    output_schema: Optional[Dict] = None
    examples: Optional[List[Dict]] = None
    tags: Optional[List[str]] = None
    image_url: Optional[str] = None


class AgentCardInDB(AgentCardBase):
    id: int
    agent_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class AgentCard(AgentCardInDB):
    pass


# Feedback schemas
class FeedbackBase(BaseModel):
    rating: int
    comments: Optional[str] = None

    @validator("rating")
    def rating_must_be_valid(cls, v):
        if v < 1 or v > 5:
            raise ValueError("Rating must be between 1 and 5")
        return v


class FeedbackCreate(FeedbackBase):
    agent_id: str
    collaboration_id: Optional[int] = None


class FeedbackUpdate(BaseModel):
    rating: Optional[int] = None
    comments: Optional[str] = None

    @validator("rating")
    def rating_must_be_valid(cls, v):
        if v is not None and (v < 1 or v > 5):
            raise ValueError("Rating must be between 1 and 5")
        return v


class FeedbackInDB(FeedbackBase):
    id: int
    user_id: int
    agent_id: str
    collaboration_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class Feedback(FeedbackInDB):
    pass


# Collaboration schemas
class CollaborationBase(BaseModel):
    initiator_agent_id: str
    collaborator_agent_id: str
    status: str = "pending"


class CollaborationCreate(CollaborationBase):
    pass


class CollaborationUpdate(BaseModel):
    status: Optional[str] = None
    input_data: Optional[Dict] = None
    output_data: Optional[Dict] = None
    negotiation_history: Optional[List[Dict]] = None
    error_message: Optional[str] = None


class CollaborationInDB(CollaborationBase):
    id: int
    input_data: Optional[Dict] = None
    output_data: Optional[Dict] = None
    negotiation_history: Optional[List[Dict]] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        orm_mode = True


class Collaboration(CollaborationInDB):
    pass


# Agent Run schemas
class AgentRunRequest(BaseModel):
    input: str
    stream: bool = False


class AgentRunResponse(BaseModel):
    output: str


# Similarity Search schemas
class SimilaritySearchRequest(BaseModel):
    agent_id: str
    top_k: int = 5
    similarity_threshold: float = 0.5
    method: str = "gnn"  # "gnn" or "jaccard"


class SimilarAgentResponse(BaseModel):
    agent_id: str
    name: str
    similarity_score: float
    agent_type: str
    capabilities: List[str]


class SimilaritySearchResponse(BaseModel):
    similar_agents: List[SimilarAgentResponse]


# Collaboration Recommendation schemas
class CollaborationRecommendationRequest(BaseModel):
    agent_id: str
    top_k: int = 5


class RecommendedAgentResponse(BaseModel):
    agent_id: str
    name: str
    success_probability: float
    agent_type: str
    capabilities: List[str]


class CollaborationRecommendationResponse(BaseModel):
    recommended_agents: List[RecommendedAgentResponse]


# Collaboration Execution schemas
class CollaborationExecuteRequest(BaseModel):
    input_data: Dict


class CollaborationExecuteResponse(BaseModel):
    collaboration_id: int
    status: str
    output_data: Optional[Dict] = None
    error_message: Optional[str] = None