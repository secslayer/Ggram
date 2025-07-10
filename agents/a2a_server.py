import asyncio
import json
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from python_a2a.models.message import Message, TextContent, MessageRole
from python_a2a.server import A2AServer
from python_a2a.models.task import Task, TaskStatus

from core.config import settings
from models.agent import Agent
from core.database import get_db


class A2AAgentServer:
    """A2A server implementation for agents."""

    def __init__(self, agent_id: str, agent_instance: Any):
        """Initialize the A2A server."""
        self.agent_id = agent_id
        self.agent_instance = agent_instance
        self.server = A2AServer()
        self.tasks = {}
        
        # Register handlers
        self.server.register_handler("create_task", self.handle_create_task)
        self.server.register_handler("get_task_status", self.handle_get_task_status)
        self.server.register_handler("get_task_result", self.handle_get_task_result)
        self.server.register_handler("cancel_task", self.handle_cancel_task)
    
    async def handle_create_task(self, message: Message) -> Dict[str, Any]:
        """Handle create task requests."""
        try:
            # Extract task data from message
            input_text = message.content.text
            
            # Create a new task
            task_id = f"task_{len(self.tasks) + 1}"
            task = Task(task_id=task_id, status=TaskStatus.PENDING)
            self.tasks[task_id] = task
            
            # Run the agent in a background task
            asyncio.create_task(self._run_agent(task_id, input_text))
            
            return {"task_id": task_id}
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
    
    async def handle_get_task_status(self, message: Message) -> Dict[str, Any]:
        """Handle get task status requests."""
        try:
            task_id = message.content.text
            if task_id not in self.tasks:
                return {"error": "Task not found"}
            
            task = self.tasks[task_id]
            return {"status": task.status.value}
        except Exception as e:
            return {"error": str(e)}
    
    async def handle_get_task_result(self, message: Message) -> Dict[str, Any]:
        """Handle get task result requests."""
        try:
            task_id = message.content.text
            if task_id not in self.tasks:
                return {"error": "Task not found"}
            
            task = self.tasks[task_id]
            if task.status != TaskStatus.COMPLETED:
                return {"error": "Task not completed"}
            
            return {"result": task.result}
        except Exception as e:
            return {"error": str(e)}
    
    async def handle_cancel_task(self, message: Message) -> Dict[str, Any]:
        """Handle cancel task requests."""
        try:
            task_id = message.content.text
            if task_id not in self.tasks:
                return {"error": "Task not found"}
            
            task = self.tasks[task_id]
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                return {"error": "Task already completed or failed"}
            
            task.status = TaskStatus.CANCELLED
            return {"status": task.status.value}
        except Exception as e:
            return {"error": str(e)}
    
    async def _run_agent(self, task_id: str, input_text: str):
        """Run the agent and update the task."""
        task = self.tasks[task_id]
        task.status = TaskStatus.RUNNING
        
        try:
            # Run the agent
            result = self.agent_instance.run(input_text)
            
            # Update the task
            task.status = TaskStatus.COMPLETED
            task.result = result
        except Exception as e:
            # Update the task with error
            task.status = TaskStatus.FAILED
            task.error = str(e)
    
    async def handle_request(self, request: Request) -> Dict[str, Any]:
        """Handle incoming HTTP requests."""
        try:
            # Parse the request body
            body = await request.json()
            
            # Create an A2A message
            message = Message(
                sender=body.get("sender", ""),
                receiver=self.agent_id,
                message_id=body.get("message_id", ""),
                content=TextContent(text=body.get("content", {}).get("text")),
                role=MessageRole.USER
            )
            
            # Process the message
            response = await self.server.process_message(message)
            
            return response
        except Exception as e:
            import traceback
            print(f"Error in handle_request: {e}")
            traceback.print_exc()
            return {"error": str(e)}


def create_a2a_endpoints(app: FastAPI):
    """Create A2A endpoints for the FastAPI app."""
    
    @app.post("/a2a/{agent_id}")
    async def a2a_endpoint(agent_id: str, request: Request, db: Session = Depends(get_db)):
        """A2A endpoint for agents."""
        # Import here to avoid circular imports
        from services.agent_service import get_agent_instance
        try:
            # Get the agent instance
            agent_instance = await get_agent_instance(agent_id, db)
            if not agent_instance:
                raise HTTPException(status_code=404, detail="Agent not found")
            # Create an A2A server for the agent
            server = A2AAgentServer(agent_id, agent_instance)
            # Handle the request
            response = await server.handle_request(request)
            return response
        except Exception as e:
            import traceback
            print(f"Error in a2a_endpoint: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))