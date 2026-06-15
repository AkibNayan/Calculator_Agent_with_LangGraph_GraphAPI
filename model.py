from langchain_groq.chat_models import ChatGroq
import os
from dotenv import load_dotenv
from tools import multiply, add, divide

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")


model = ChatGroq(model="qwen/qwen3-32b", api_key=groq_api_key, temperature=0)

tools = [multiply, add, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)
