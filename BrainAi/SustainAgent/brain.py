#from dotenv import load_dotenv
import json

#load_dotenv()
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.tools import FunctionTool


llm = Ollama(model="llama3:latest", request_timeout=300.0)
Settings.llm=llm

def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)


def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b


add_tool = FunctionTool.from_defaults(fn=add)

agent = ReActAgent.from_tools([multiply_tool, add_tool], llm=llm, verbose=True)
response = agent.chat("What is 20+(2*4)? Use a tool to calculate every step.")