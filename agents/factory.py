from typing import Any, Dict, List, Optional, Type, Union

from langchain.agents import AgentExecutor, AgentType, initialize_agent
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from .base import BaseAgent


class SimpleAgent(BaseAgent):
    """Simple agent that uses a basic LLMChain."""

    def _init_agent(self) -> LLMChain:
        """Initialize a simple LLMChain agent."""
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


class ToolAgent(BaseAgent):
    """Agent that uses tools via the LangChain agent framework."""

    def _init_agent(self) -> AgentExecutor:
        """Initialize an agent with tools."""
        if not self.tools:
            raise ValueError("Tools are required for a ToolAgent")
        
        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            agent_kwargs={
                "prefix": f"You are {self.name}, {self.description}",
            },
        )


class RAGAgent(BaseAgent):
    """Agent that uses Retrieval-Augmented Generation (RAG)."""

    def __init__(
        self,
        name: str,
        description: str,
        retriever: Any,  # Vector store retriever
        **kwargs,
    ):
        """Initialize a RAG agent."""
        self.retriever = retriever
        super().__init__(name, description, **kwargs)
    
    def _init_agent(self) -> LLMChain:
        """Initialize a RAG agent."""
        if not self.prompt_template:
            self.prompt_template = (
                "You are {name}, {description}\n"
                "Use the following context to answer the question:\n"
                "{context}\n\n"
                "Human: {input}\n"
                "AI: "
            )
        
        prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["input", "context"],
            partial_variables={
                "name": self.name,
                "description": self.description,
            },
        )
        
        return LLMChain(llm=self.llm, prompt=prompt, memory=self.memory)
    
    def run(self, input_text: str) -> str:
        """Run the RAG agent with the given input."""
        # Retrieve relevant documents
        docs = self.retriever.get_relevant_documents(input_text)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Run the agent with the context
        return self.agent.run(input=input_text, context=context)


class GraphAgent(BaseAgent):
    """Agent that uses a LangChain graph of chains."""

    def __init__(
        self,
        name: str,
        description: str,
        chains: List[LLMChain],
        **kwargs,
    ):
        """Initialize a graph agent."""
        self.chains = chains
        super().__init__(name, description, **kwargs)
    
    def _init_agent(self) -> SequentialChain:
        """Initialize a graph agent with sequential chains."""
        if not self.chains:
            raise ValueError("Chains are required for a GraphAgent")
        
        # Create a sequential chain from the provided chains
        return SequentialChain(
            chains=self.chains,
            input_variables=["input"],
            output_variables=["output"],
            verbose=True,
        )
    
    def run(self, input_text: str) -> str:
        """Run the graph agent with the given input."""
        result = self.agent.run(input=input_text)
        return result["output"] if isinstance(result, dict) else result


class AgentFactory:
    """Factory for creating different types of agents."""

    @staticmethod
    def create_agent(
        agent_type: str,
        name: str,
        description: str,
        **kwargs,
    ) -> BaseAgent:
        """Create an agent of the specified type."""
        if agent_type == "simple":
            return SimpleAgent(name, description, **kwargs)
        elif agent_type == "tool":
            return ToolAgent(name, description, **kwargs)
        elif agent_type == "rag":
            if "retriever" not in kwargs:
                raise ValueError("Retriever is required for a RAG agent")
            return RAGAgent(name, description, **kwargs.pop("retriever"), **kwargs)
        elif agent_type == "graph":
            if "chains" not in kwargs:
                raise ValueError("Chains are required for a Graph agent")
            return GraphAgent(name, description, **kwargs.pop("chains"), **kwargs)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")