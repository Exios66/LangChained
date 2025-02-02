import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from langsmith import traceable

# ======================
# 1. Environment Setup
# ======================

def check_environment():
    """Verify all required API keys are present"""
    required_keys = {
        'TAVILY_API_KEY': 'Tavily Search API',
        'ANTHROPIC_API_KEY': 'Anthropic Claude API',
        'LANGCHAIN_API_KEY': 'LangSmith API'
    }
    
    missing_keys = []
    for key, service in required_keys.items():
        if not os.getenv(key):
            missing_keys.append(f"{service} ({key})")
    
    if missing_keys:
        raise EnvironmentError(
            "Missing required API keys:\n" + 
            "\n".join(f"- {key}" for key in missing_keys) +
            "\n\nPlease set these environment variables or add them to your .env file"
        )

# Load environment variables from .env file if present
load_dotenv()

# Verify environment before proceeding
check_environment()

# ======================
# 2. Framework Setup
# ======================

# Initialize core components with explicit API keys
llm = ChatAnthropic(
    model="claude-3-opus-20240229",
    anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')
)

search_tool = TavilySearchResults(
    max_results=3,
    tavily_api_key=os.getenv('TAVILY_API_KEY')
)

# ======================
# 2. State Definition
# ======================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_actor: str
    research_data: dict

# ======================
# 3. Agent Definitions
# ======================

class ResearchAgent:
    @traceable(name="Research Agent")
    def __call__(self, state: AgentState):
        last_message = state["messages"][-1].content
        research_result = search_tool.invoke({"query": last_message})
        return {
            "messages": [HumanMessage(content=f"Research data: {research_result}")],
            "research_data": research_result,
            "current_actor": "analyst"
        }

class AnalysisAgent:
    @traceable(name="Analysis Agent")
    def __call__(self, state: AgentState):
        analysis = llm.invoke(f"Analyze this data: {state['research_data']}")
        return {
            "messages": [HumanMessage(content=analysis.content)],
            "current_actor": "writer"
        }

class WritingAgent:
    @traceable(name="Writing Agent")
    def __call__(self, state: AgentState):
        final_response = llm.invoke(
            f"Generate final response using: {state['messages'][-1].content}"
        )
        return {
            "messages": [final_response],
            "current_actor": "end"
        }

# ======================
# 4. Graph Construction
# ======================

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("researcher", ResearchAgent())
workflow.add_node("analyst", AnalysisAgent())
workflow.add_node("writer", WritingAgent())

# Set entry point
workflow.set_entry_point("researcher")

# Add conditional edges
def route_based_on_role(state: AgentState):
    return state["current_actor"]

workflow.add_conditional_edges(
    "researcher",
    route_based_on_role,
    {"analyst": "analyst", "writer": "writer", "end": END}
)

workflow.add_conditional_edges(
    "analyst",
    route_based_on_role,
    {"writer": "writer", "end": END}
)

workflow.add_conditional_edges(
    "writer",
    route_based_on_role,
    {"end": END}
)

# Compile the workflow
app = workflow.compile()

# ======================
# 5. Execution Example
# ======================

if __name__ == "__main__":
    inputs = {
        "messages": [HumanMessage(content="Research latest AI advancements")],
        "current_actor": "researcher",
        "research_data": {}
    }
    
    for output in app.stream(inputs):
        current_state = output.keys()
        print(f"Current state: {current_state}")
        print("---" * 20) 