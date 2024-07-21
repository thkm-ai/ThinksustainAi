from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import json
from dotenv import load_dotenv
load_dotenv()
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.tools import FunctionTool

#define model settings
llm = Ollama(model="llama3:latest", request_timeout=300.0)
Settings.llm=llm

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

#################################################################

documents = SimpleDirectoryReader("/home/azureuser/Brain/BrainAi/SustainAgent/storage").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()


def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b

def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b


multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)

green_tool = QueryEngineTool.from_defaults(
    query_engine,
    name="european_green_deal",
    description="A RAG engine with some basic facts about Green Deal.",
)


agent = ReActAgent.from_tools([multiply_tool,add_tool,
     green_tool], verbose=True
)

response = agent.chat(
    "From the Financing the Transformation,pick the European Commission sustainable investment euro estimation in trillions and multiplied by 3"
)

print(response)

#response = query_engine.query(
#    "What is the GreenDeal"
#)
#print(response)