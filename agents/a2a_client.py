import asyncio
import json
from typing import Any, Dict, List, Optional, Union

import aiohttp
from python_a2a.models.message import Message, TextContent, MessageRole
from python_a2a.models.task import Task, TaskStatus


class A2AAgentClient:
    """A2A client implementation for agent-to-agent communication."""

    def __init__(self, sender_id: str, receiver_id: str, receiver_endpoint: str):
        """Initialize the A2A client."""
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.receiver_endpoint = receiver_endpoint
    
    async def create_task(self, input_text: str) -> str:
        """Create a task on the receiver agent."""
        message = Message(
            message_id="create_task",
            content=TextContent(text=input_text),
            role=MessageRole.USER,
        )
        message.sender = self.sender_id
        message.receiver = self.receiver_id
        
        response = await self._send_message(message)
        if "error" in response:
            raise Exception(response["error"])
        
        return response["task_id"]
    
    async def get_task_status(self, task_id: str) -> str:
        """Get the status of a task."""
        message = Message(
            message_id="get_task_status",
            content=TextContent(text=task_id),
            role=MessageRole.USER,
        )
        message.sender = self.sender_id
        message.receiver = self.receiver_id
        
        response = await self._send_message(message)
        if "error" in response:
            raise Exception(response["error"])
        
        return response["status"]
    
    async def get_task_result(self, task_id: str) -> Any:
        """Get the result of a completed task."""
        message = Message(
            message_id="get_task_result",
            content=TextContent(text=task_id),
            role=MessageRole.USER,
        )
        message.sender = self.sender_id
        message.receiver = self.receiver_id
        
        response = await self._send_message(message)
        if "error" in response:
            raise Exception(response["error"])
        
        return response["result"]
    
    async def cancel_task(self, task_id: str) -> str:
        """Cancel a task."""
        message = Message(
            message_id="cancel_task",
            content=TextContent(text=task_id),
            role=MessageRole.USER,
        )
        message.sender = self.sender_id
        message.receiver = self.receiver_id
        
        response = await self._send_message(message)
        if "error" in response:
            raise Exception(response["error"])
        
        return response["status"]
    
    async def _send_message(self, message: Message) -> Dict[str, Any]:
        """Send a message to the receiver agent."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.receiver_endpoint,
                json=message.to_dict(),
            ) as response:
                if response.status != 200:
                    return {"error": f"HTTP error: {response.status}"}
                
                return await response.json()
    
    async def run_task(self, input_text: str, polling_interval: float = 0.5, timeout: float = 30.0) -> Any:
        """Run a task and wait for the result."""
        # Create the task
        print(f"Creating task with input: {input_text}")
        task_id = await self.create_task(input_text)
        print(f"Task created with ID: {task_id}")
        
        # Poll for task completion
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < timeout:
            print(f"Getting status for task: {task_id}")
            status = await self.get_task_status(task_id)
            print(f"Task status: {status}")
            
            if status == TaskStatus.COMPLETED.value:
                return await self.get_task_result(task_id)
            elif status in [TaskStatus.FAILED.value, TaskStatus.CANCELLED.value]:
                raise Exception(f"Task {task_id} {status}")
            
            await asyncio.sleep(polling_interval)
        
        # Timeout
        await self.cancel_task(task_id)
        raise Exception(f"Task {task_id} timed out")