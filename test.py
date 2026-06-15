from langgraph.graph import StateGraph, START, END
from state import MessagesState
from model_node import llm_call
from tool_node import tool_node
from end_logic import should_continue
from IPython.display import display, Image
from langchain.messages import HumanMessage

agent_builder = StateGraph(MessagesState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()

# Show the agent
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))


# Invoke
messages = [HumanMessage(content="Add 3 and 4.")]
results = agent.invoke({"messages": messages})

for msg in results["messages"]:
    msg.pretty_print()
