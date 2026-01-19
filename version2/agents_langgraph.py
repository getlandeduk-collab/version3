"""
Proper LangGraph implementation with all required features:
- Memory management (checkpointing)
- Tool integration (proper tool execution)
- Planning & decision making (conditional routing)
- Multi-step workflows (StateGraph)
- Debugging & visualization (state inspection)
- Reusable components/policies
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Union, Literal
import json
import re
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict, Annotated
import operator

# Import existing tools
from agents import FirecrawlTool, get_model_config


# ==================== STATE MANAGEMENT ====================

class AgentState(TypedDict):
    """State schema for LangGraph agents."""
    messages: Annotated[List[BaseMessage], add_messages]
    candidate_profile: Optional[Dict[str, Any]]
    job_details: Optional[Dict[str, Any]]
    match_score: Optional[float]
    summary: Optional[str]
    current_step: str
    workflow_data: Dict[str, Any]


# ==================== MEMORY MANAGEMENT ====================

def create_memory_store():
    """Create a memory checkpoint store for conversation history."""
    return MemorySaver()


# ==================== REUSABLE NODE COMPONENTS ====================

def create_agent_node(
    name: str,
    role: str,
    instructions: List[str],
    model: ChatOpenAI,
    tools: List[BaseTool] = None
):
    """Create a reusable agent node for LangGraph."""
    
    def agent_node(state: AgentState) -> Dict[str, Any]:
        """Node function that processes state and returns updated state."""
        # Build system prompt
        system_prompt = f"Role: {role}\n\n" + "\n".join(instructions)
        
        # Get messages from state
        messages = state.get("messages", [])
        
        # Add system message if not present
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=system_prompt)] + messages
        
        # Use model with tools if available
        if tools:
            model_with_tools = model.bind_tools(tools)
            response = model_with_tools.invoke(messages)
        else:
            response = model.invoke(messages)
        
        # Update state with response
        return {
            "messages": [response],
            "current_step": name
        }
    
    return agent_node


def create_tool_node(tools: List[BaseTool]) -> ToolNode:
    """Create a tool execution node."""
    if not tools:
        raise ValueError("Tools list cannot be empty for ToolNode")
    return ToolNode(tools)


def create_decision_node(condition_func):
    """Create a conditional routing node."""
    def decision_node(state: AgentState) -> Literal["continue", "end", "retry"]:
        return condition_func(state)
    return decision_node


# ==================== PLANNING & DECISION MAKING ====================

def should_continue_workflow(state: AgentState) -> Literal["tools", "end"]:
    """Decision function: continue with tools or end."""
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "end"


def should_summarize(state: AgentState) -> Literal["summarize", "skip"]:
    """Decision function: should we summarize this job match?"""
    match_score = state.get("match_score", 0.0)
    if match_score >= 0.5:
        return "summarize"
    return "skip"


# ==================== MULTI-STEP WORKFLOW GRAPH ====================

def create_job_matching_graph(
    model_name: str = "gpt-4o",
    firecrawl_api_key: str = None
) -> StateGraph:
    """
    Create a complete job matching workflow graph with:
    - Resume parsing
    - Job scraping
    - Job scoring
    - Conditional summarization
    """
    
    # Initialize models
    parser_model = get_model_config(model_name, default_temperature=0)
    scraper_model = get_model_config(model_name, default_temperature=0)
    scorer_model = get_model_config(model_name, default_temperature=0)
    summarizer_model = get_model_config(model_name, default_temperature=0.3)
    
    # Initialize tools
    firecrawl_tool = FirecrawlTool(api_key=firecrawl_api_key)
    
    # Create nodes
    resume_parser_node = create_agent_node(
        name="Resume Parser",
        role="Extract and structure all information from resume OCR text",
        instructions=[
            "Parse raw OCR text from resume and extract ALL information.",
            "You MUST return ONLY valid JSON (no markdown, no code blocks, no explanations).",
            "Extract: name, email, phone, skills, experience_summary, total_years_experience, education, certifications, interests",
        ],
        model=parser_model
    )
    
    job_scraper_node = create_agent_node(
        name="Job Scraper",
        role="Extract complete job posting information from URLs",
        instructions=[
            "Given a job URL, extract ALL available information.",
            "Extract: job_title, company_name, description, required_skills, required_experience, qualifications, responsibilities, salary, location, job_type",
        ],
        model=scraper_model,
        tools=[firecrawl_tool]
    )
    
    job_scorer_node = create_agent_node(
        name="Job Scorer",
        role="Evaluate candidate-job match accurately",
        instructions=[
            "Analyze candidate profile and job details to calculate match score.",
            "Return JSON with: match_score, key_matches, requirements_met, total_requirements, reasoning, mismatch_areas",
        ],
        model=scorer_model
    )
    
    summarizer_node = create_agent_node(
        name="Summarizer",
        role="Generate accurate, unique summaries",
        instructions=[
            "Write a unique 150-200 word summary based on match_score and job details.",
            "Include: fit assessment, matching skills, gaps, growth opportunities, practical considerations",
        ],
        model=summarizer_model
    )
    
    # Tool execution node
    tool_node = create_tool_node([firecrawl_tool])
    
    # Create graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("resume_parser", resume_parser_node)
    workflow.add_node("job_scraper", job_scraper_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("job_scorer", job_scorer_node)
    workflow.add_node("summarizer", summarizer_node)
    
    # Add edges
    workflow.set_entry_point("resume_parser")
    workflow.add_edge("resume_parser", "job_scraper")
    
    # Conditional routing: tools or continue
    workflow.add_conditional_edges(
        "job_scraper",
        should_continue_workflow,
        {
            "tools": "tools",
            "end": "job_scorer"
        }
    )
    workflow.add_edge("tools", "job_scraper")  # Loop back after tools
    
    # Conditional routing: summarize or skip
    workflow.add_conditional_edges(
        "job_scorer",
        should_summarize,
        {
            "summarize": "summarizer",
            "skip": END
        }
    )
    workflow.add_edge("summarizer", END)
    
    return workflow.compile(checkpointer=create_memory_store())


# ==================== DEBUGGING & VISUALIZATION ====================

def visualize_graph(graph: StateGraph):
    """Visualize the graph structure."""
    try:
        # Try to use LangGraph's built-in visualization
        import networkx as nx
        import matplotlib.pyplot as plt
        
        # Get graph structure
        graph_structure = graph.get_graph()
        
        # Create NetworkX graph
        G = nx.DiGraph()
        for node in graph_structure.nodes:
            G.add_node(node)
        for edge in graph_structure.edges:
            G.add_edge(edge.source, edge.target)
        
        # Visualize
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=2000, font_size=10, font_weight='bold', arrows=True)
        plt.title("Job Matching Workflow Graph")
        plt.show()
        
        return "Graph visualization displayed"
    except ImportError:
        return "Install networkx and matplotlib for visualization: pip install networkx matplotlib"


def inspect_state(state: AgentState) -> Dict[str, Any]:
    """Inspect and return state information for debugging."""
    return {
        "current_step": state.get("current_step"),
        "message_count": len(state.get("messages", [])),
        "has_candidate_profile": state.get("candidate_profile") is not None,
        "has_job_details": state.get("job_details") is not None,
        "match_score": state.get("match_score"),
        "has_summary": state.get("summary") is not None,
        "workflow_data_keys": list(state.get("workflow_data", {}).keys())
    }


# ==================== ENHANCED AGENT CLASS WITH LANGGRAPH ====================

class LangGraphAgent:
    """Proper LangGraph-based agent with all features."""
    
    def __init__(
        self,
        name: str,
        role: str,
        instructions: List[str],
        model: ChatOpenAI,
        tools: List[BaseTool] = None,
        enable_memory: bool = True
    ):
        self.name = name
        self.role = role
        self.instructions = instructions
        self.model = model
        self.tools = tools or []
        self.enable_memory = enable_memory
        
        # Create graph
        self.graph = self._create_agent_graph()
    
    def _create_agent_graph(self) -> StateGraph:
        """Create a graph for this agent."""
        workflow = StateGraph(AgentState)
        
        # Add agent node
        agent_node = create_agent_node(
            self.name,
            self.role,
            self.instructions,
            self.model,
            self.tools
        )
        workflow.add_node("agent", agent_node)
        
        # Add tool node if tools available
        if self.tools:
            tool_node = create_tool_node(self.tools)
            workflow.add_node("tools", tool_node)
            
            # Conditional routing
            workflow.set_entry_point("agent")
            workflow.add_conditional_edges(
                "agent",
                should_continue_workflow,
                {
                    "tools": "tools",
                    "end": END
                }
            )
            workflow.add_edge("tools", "agent")
        else:
            workflow.set_entry_point("agent")
            workflow.add_edge("agent", END)
        
        # Compile with memory if enabled
        if self.enable_memory:
            return workflow.compile(checkpointer=create_memory_store())
        else:
            return workflow.compile()
    
    def run(
        self,
        input_data: Union[str, Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run the agent with input data."""
        # Prepare initial state
        if isinstance(input_data, dict):
            input_text = json.dumps(input_data, indent=2)
        else:
            input_text = str(input_data)
        
        initial_state = {
            "messages": [HumanMessage(content=input_text)],
            "current_step": self.name,
            "workflow_data": {}
        }
        
        # Run graph
        config = config or {}
        if self.enable_memory:
            config["configurable"] = {"thread_id": config.get("thread_id", "default")}
        
        result = self.graph.invoke(initial_state, config=config)
        
        # Extract response
        messages = result.get("messages", [])
        last_message = messages[-1] if messages else None
        
        if isinstance(last_message, AIMessage):
            return {
                "content": last_message.content,
                "tool_calls": last_message.tool_calls if hasattr(last_message, 'tool_calls') else [],
                "state": result
            }
        
        return {
            "content": str(last_message) if last_message else "",
            "state": result
        }
    
    def visualize(self):
        """Visualize the agent's graph."""
        return visualize_graph(self.graph)
    
    def inspect(self, state: AgentState):
        """Inspect agent state."""
        return inspect_state(state)


# ==================== BACKWARD COMPATIBILITY WRAPPER ====================

def create_langgraph_agent(
    name: str = None,
    role: str = None,
    model: ChatOpenAI = None,
    instructions: List[str] = None,
    tools: List[BaseTool] = None,
    show_tool_calls: bool = True,
    markdown: bool = True,
    response_format: Dict[str, Any] = None
) -> LangGraphAgent:
    """Create a LangGraph agent (replacement for old Agent class)."""
    return LangGraphAgent(
        name=name or "Agent",
        role=role or "",
        instructions=instructions or [],
        model=model,
        tools=tools or [],
        enable_memory=True
    )

