# Ggram - Collaborative Agentic AI Platform

This project aims to build a platform where users can create, share, and collaborate with AI agents. The system leverages several advanced technologies to enable intelligent agent interaction and discovery.

## Core Features

- **User Accounts**: Users can sign up and manage their profiles.
- **Agent Creation**: Users can create custom AI agents using LangChain, with NVIDIA models as the backend LLM.
- **Agent-to-Agent (A2A) Communication**: Agents communicate and negotiate using the Google A2A protocol.
- **Similarity Search**: A Graph Neural Network (GNN) based similarity search helps users find agents with complementary capabilities for collaboration.
- **Graph Reinforcement Learning (GRL)**: The overall network of collaborating agents is optimized using GRL, with user feedback (likes/dislikes on agent performance) as the reward signal.
- **Agent Cards**: Each agent is represented by a card detailing its inputs, outputs, and capabilities, which can be shared and rated by the community.

## Project Structure

```
/ggram
|-- api/                # FastAPI endpoints and routes
|-- core/               # Core application logic, config, DB session
|-- agents/             # Agent creation, A2A server logic
|-- models/             # SQLAlchemy database models
|-- services/           # Business logic (similarity search, RL)
|-- static/             # Frontend assets (HTML, CSS, JS)
|-- templates/          # Jinja2 templates for frontend
|-- migrations/         # Alembic database migrations
|-- .env                # Environment variables (API keys, etc.)
|-- alembic.ini         # Alembic configuration
|-- main.py             # Main application entry point
|-- requirements.txt    # Python dependencies
|-- README.md           # This file
```

## Getting Started

### Prerequisites

- Python 3.9+
- pip
- Virtual environment (recommended)

### Setup Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/secslayer/Ggram.git
   cd ggram
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configure Environment Variables

1. Create a `.env` file in the project root (or copy the example):
   ```bash
   cp .env.example .env  # If using the example file
   ```

2. Edit the `.env` file and add your NVIDIA API key and other settings:
   ```
   # API Keys
   NVIDIA_API_KEY="nvapi-your-key-here"
   
   # Database Configuration
   DATABASE_URL="sqlite:///./ggram.db"
   
   # Security
   SECRET_KEY="your-secret-key-for-jwt-tokens"
   ALGORITHM="HS256"
   ACCESS_TOKEN_EXPIRE_MINUTES=30
   
   # LLM Settings
   LLM_MODEL="meta/llama-3.1-8b-instruct"
   LLM_TEMPERATURE=0.2
   LLM_TOP_P=0.7
   LLM_MAX_TOKENS=1024
   
   # A2A Protocol Settings
   A2A_SERVER_HOST="0.0.0.0"
   A2A_SERVER_PORT=8000
   ```

### Database Setup

1. Initialize the database with Alembic:
   ```bash
   alembic revision --autogenerate -m "Initial migration"
   alembic upgrade head
   ```

### Running the Application

1. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

2. Access the application:
   - Web Interface: http://localhost:8000
   - API Documentation: http://localhost:8000/api/docs
   - OpenAPI Spec: http://localhost:8000/api/openapi.json

## Development

### Code Formatting

This project uses Black for code formatting and isort for import sorting:

```bash
black .
isort .
```

### Linting

Use flake8 for linting:

```bash
flake8 .
```

### Database Migrations

When you make changes to the database models, create a new migration:

```bash
alembic revision --autogenerate -m "Description of changes"
alembic upgrade head
```

## API Documentation

The API documentation is automatically generated using FastAPI's built-in Swagger UI and ReDoc:

- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

## License

This project is licensed under the MIT License - see the LICENSE file for details.
