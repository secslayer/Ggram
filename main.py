import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html

from api.endpoints import users, auth, agents, agent_cards, feedback, collaborations, similarity
from agents.a2a_server import create_a2a_endpoints
from core.config import settings

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    docs_url=None,  # Disable default docs URL
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Include API routers
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(users.router, prefix="/api/users", tags=["users"])
app.include_router(agents.router, prefix="/api/agents", tags=["agents"])
app.include_router(agent_cards.router, prefix="/api/agent-cards", tags=["agent-cards"])
app.include_router(feedback.router, prefix="/api/feedback", tags=["feedback"])
app.include_router(collaborations.router, prefix="/api/collaborations", tags=["collaborations"])
app.include_router(similarity.router, prefix="/api/similarity", tags=["similarity"])

# Create A2A endpoints
create_a2a_endpoints(app)

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "settings": settings})

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": settings.app_version}

# Custom API docs endpoint
@app.get("/api/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=f"{settings.app_name} - API Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )

# Run the application
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host=settings.a2a_server_host,
        port=port,
        reload=True,
    )