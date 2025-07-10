from typing import Any, Dict, List, Optional, Union

from langchain.agents import AgentExecutor
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from core.config import settings


class BaseAgent:
    """Base agent class for creating agents with LangChain."""

    def __init__(
        self,
        name: str,
        description: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        prompt_template: Optional[str] = None,
        tools: Optional[List[Any]] = None,
    ):
        """Initialize the base agent."""
        self.name = name
        self.description = description
        
        # LLM settings
        self.model = model or settings.llm_model
        self.temperature = temperature or settings.llm_temperature
        self.top_p = top_p or settings.llm_top_p
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.api_key = api_key or settings.nvidia_api_key
        
        # Agent settings
        self.prompt_template = prompt_template
        self.tools = tools or []
        
        # Initialize LLM
        self.llm = self._init_llm()
        
        # Initialize memory
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Initialize agent
        self.agent = self._init_agent()
    
    def _init_llm(self) -> ChatNVIDIA:
        """Initialize the LLM."""
        return ChatNVIDIA(
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
    
    def _init_agent(self) -> Union[LLMChain, AgentExecutor]:
        """Initialize the agent.
        
        This method should be overridden by subclasses to create specific types of agents.
        """
        if not self.prompt_template:
            self.prompt_template = (
                "You are {name}, {description}\n"
                "Human: {input}\n"
                "AI: "
            )
        
        prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["input"],
            partial_variables={
                "name": self.name,
                "description": self.description,
            },
        )
        
        return LLMChain(llm=self.llm, prompt=prompt, memory=self.memory)
    
    def run(self, input_text: str) -> str:
        """Run the agent with the given input."""
        return self.agent.run(input_text)
    
    def stream(self, input_text: str):
        """Stream the agent's response."""
        if hasattr(self.agent, "stream"):
            return self.agent.stream(input_text)
        else:
            # Fallback for agents that don't support streaming
            result = self.run(input_text)
            yield result