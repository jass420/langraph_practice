import os, getpass
from langchain_openai import ChatOpenAI
from pprint import pprint
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from IPython.display import Image, display
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
import sqlite3
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from dotenv import load_dotenv
load_dotenv()
# Retrieve OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("API key not found. Ensure your .env file is set correctly.")

llm = ChatOpenAI(model_name="gpt-4", temperature=0.7, openai_api_key=openai_api_key)
conn = sqlite3.connect(":memory:", check_same_thread = False)

def set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Enter {var}: ")

set_env("OPENAI_API_KEY")
set_env("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langchain-academy"

def chat_model_node(state: MessagesState):
    return {"messages": llm.invoke(state["messages"])}

class State(MessagesState):
    summary: str

def call_model(state: State):
    
    # Get summary if it exists
    summary = state.get("summary", "")

    # If there is summary, then we add it
    if summary:
        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"
        # Append summary to any newer messages
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]
    
    response = llm.invoke(messages)
    return {"messages": response}

#The two functions below are used to summarise the conversation and the last function is used to check if ewe need a summary or we can end
def summarize_conversation(state: State):
    
    # First, we get any existing summary
    summary = state.get("summary", "")

    # Create our summarization prompt 
    if summary:
        
        # A summary already exists
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
        
    else:
        summary_message = "Create a summary of the conversation above:"

#Checking if we need to END or summarise converation
#Conversation is summarised if there are more thsn 6 messages
def should_continue(state: State):
    
    """Return the next node to execute."""
    
    messages = state["messages"]
    
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"
    
    # Otherwise we can just end
    return END

# Define a new graph
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node("summarize conversation", summarize_conversation)

#Add edges
workflow.add_edge(START, "conversation")
#Putting in the edge that checks if there needs to be a summary
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize conversation", END)

#Compile
memeory = MemorySaver()
graph = workflow.compile(checkpointer = memeory)
display(Image(graph.get_graph().draw_mermaid_png()))

#Creating threads
config = {"configurable": {"thread_id": "1"}}

#Start conversation
input_message = HumanMessage(content="Hello I am Jas")
output = graph.invoke({"messages": [input_message]}, config)
for m in output['messages'][-1:]:
    m.pretty_print()
