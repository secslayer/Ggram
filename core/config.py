import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()


class Settings(BaseModel):
    """Application settings."""

    # API Keys
    nvidia_api_key: str = Field(
        default=os.getenv("NVIDIA_API_KEY", ""),
        description="NVIDIA API key for LLM access",
    )

    # Database
    database_url: str = Field(
        default=os.getenv("DATABASE_URL", "sqlite:///./ggram.db"),
        description="Database connection string",
    )

    # Security
    secret_key: str = Field(
        default=os.getenv("SECRET_KEY", "your-secret-key-for-jwt-tokens"),
        description="Secret key for JWT token generation",
    )
    algorithm: str = Field(
        default=os.getenv("ALGORITHM", "HS256"),
        description="Algorithm for JWT token generation",
    )
    access_token_expire_minutes: int = Field(
        default=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")),
        description="Minutes until access token expires",
    )

    # LLM Settings
    llm_model: str = Field(
        default=os.getenv("LLM_MODEL", "meta/llama-3.1-8b-instruct"),
        description="Default LLM model to use",
    )
    llm_temperature: float = Field(
        default=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        description="Temperature for LLM generation",
    )
    llm_top_p: float = Field(
        default=float(os.getenv("LLM_TOP_P", "0.7")),
        description="Top-p sampling for LLM generation",
    )
    llm_max_tokens: int = Field(
        default=int(os.getenv("LLM_MAX_TOKENS", "1024")),
        description="Maximum tokens for LLM generation",
    )

    # A2A Protocol Settings
    a2a_server_host: str = Field(
        default=os.getenv("A2A_SERVER_HOST", "0.0.0.0"),
        description="Host for A2A server",
    )
    a2a_server_port: int = Field(
        default=int(os.getenv("A2A_SERVER_PORT", "8000")),
        description="Port for A2A server",
    )

    # Application Settings
    app_name: str = Field(
        default="Ggram",
        description="Application name",
    )
    app_description: str = Field(
        default="Collaborative Agentic AI Platform",
        description="Application description",
    )
    app_version: str = Field(
        default="0.1.0",
        description="Application version",
    )


# Create global settings object
settings = Settings()