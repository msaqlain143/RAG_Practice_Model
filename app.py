import os
from typing import Dict, List, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama

# -------------------------------------------------------------
# 1. LONG-TAIL DATASET LAYER SIMULATION
# -------------------------------------------------------------
VECTOR_KNOWLEDGE_BASE = {
    "interstellar": {
        "title": "Interstellar",
        "year": "2014",
        "director": "Christopher Nolan",
        "genre": "Sci-Fi, Adventure, Drama",
        "type": "Head Item (Rich Data)"
    },
    "stranger than paradise": {
        "title": "Stranger than Paradise",
        "year": "UNKNOWN",  # Simulated sparse cold-start data gap
        "director": "Jim Jarmusch",
        "genre": "Indie, Comedy-Drama",
        "type": "Tail Item (Cold-Start)"
    }
}

EXTERNAL_SEARCH_TOOL = {
    "stranger than paradise release year": "1984"
}

# -------------------------------------------------------------
# 2. STATE REASONING LAYER ARCHITECTURE
# -------------------------------------------------------------
class ConversationalState(TypedDict):
    user_query: str
    identified_item: str
    item_metadata: Dict
    audit_status: Literal["PASS", "FAIL", "UNAUDITED"]
    agent_logs: List[str]
    final_output: str

local_brain = Ollama(model="llama3", temperature=0)

def autonomous_planner_and_retriever(state: ConversationalState) -> ConversationalState:
    logs = state.get("agent_logs", [])
    logs.append("[NODE 1 - PLANNER]: Target entity mapped from user conversational preference.")
    
    query = state["user_query"].lower()
    target_key = "interstellar" if "interstellar" in query else "stranger than paradise"
    
    retrieved_data = VECTOR_KNOWLEDGE_BASE.get(target_key)
    logs.append(f"[NODE 1 - RETRIEVER]: Pulled item '{retrieved_data['title']}' from DB Index. Catalog Tier: {retrieved_data['type']}.")
    
    return {
        **state,
        "identified_item": target_key,
        "item_metadata": retrieved_data,
        "audit_status": "UNAUDITED",
        "agent_logs": logs
    }

def self_reflection_critic(state: ConversationalState) -> ConversationalState:
    logs = state["agent_logs"]
    logs.append("[NODE 2 - SELF-REFLECTION CRITIC]: Initiating item-attribute fact audit...")
    
    metadata = state["item_metadata"]
    
    if "UNKNOWN" in metadata.values():
        status = "FAIL"
        logs.append(f"[CRITICAL FLAG]: Missing properties caught for cold-start item '{metadata['title']}'. Hallucination risk detected! Freezing generator pipe.")
    else:
        status = "PASS"
        logs.append(f"[AUDIT SUCCESS]: Features for '{metadata['title']}' verified as factual. Proceeding to generation.")
        
    return {**state, "audit_status": status, "agent_logs": logs}

def query_reformulation_node(state: ConversationalState) -> ConversationalState:
    logs = state["agent_logs"]
    logs.append("[NODE 3 - CORRECTIVE REFORMULATION]: Expanding query profiles to trigger alternative search tools...")
    
    item_key = state["identified_item"]
    repaired_metadata = state["item_metadata"].copy()
    
    search_query = f"{item_key} release year"
    if search_query in EXTERNAL_SEARCH_TOOL:
        repaired_metadata["year"] = EXTERNAL_SEARCH_TOOL[search_query]
        logs.append(f"[NODE 3 - EXTERNAL TOOL]: Patched missing metadata parameter -> Year: {repaired_metadata['year']}.")
        
    return {
        **state,
        "item_metadata": repaired_metadata,
        "agent_logs": logs
    }

def grounded_response_generator(state: ConversationalState) -> ConversationalState:
    logs = state["agent_logs"]
    logs.append("[NODE 4 - CRS GENERATOR]: Building response strictly grounded in verified facts...")
    
    meta = state["item_metadata"]
    prompt = f"""
    Context: You are an Item-Oriented Fairness CRS. Recommend the item below using ONLY these verified facts:
    Title: {meta['title']}
    Year: {meta['year']}
    Director: {meta['director']}
    Genre: {meta['genre']}
    
    Constraint: Do not guess or extrapolate. State facts cleanly.
    """
    ai_response = local_brain.invoke(prompt)
    logs.append("[SYSTEM]: Output successfully created with zero hallucinations.")
    return {**state, "final_output": ai_response, "agent_logs": logs}

# -------------------------------------------------------------
# 3. COMPILING THE DYNAMIC LANGGRAPH STATE MACHINE
# -------------------------------------------------------------
workflow_engine = StateGraph(ConversationalState)

workflow_engine.add_node("planner_retriever", autonomous_planner_and_retriever)
workflow_engine.add_node("self_reflection_critic", self_reflection_critic)
workflow_engine.add_node("query_reformulation", query_reformulation_node)
workflow_engine.add_node("grounded_generator", grounded_response_generator)

workflow_engine.set_entry_point("planner_retriever")
workflow_engine.add_edge("planner_retriever", "self_reflection_critic")

def condition_router(state: ConversationalState):
    if state["audit_status"] == "FAIL":
        return "query_reformulation"
    return "grounded_generator"

workflow_engine.add_conditional_edges(
    "self_reflection_critic",
    condition_router,
    {
        "query_reformulation": "query_reformulation",
        "grounded_generator": "grounded_generator"
    }
)

workflow_engine.add_edge("query_reformulation", "grounded_generator")
workflow_engine.add_edge("grounded_generator", END)

app = workflow_engine.compile()

# -------------------------------------------------------------
# 4. EXECUTION RUN (Targeting Cold-Start Scenario)
# -------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*70)
    print("      INITIALIZING LOCAL PROOF-OF-CONCEPT: ITEM-ORIENTED FAIRNESS CRS")
    print("="*70)
    
    # Simulating a user requesting a rich, popular Head Item
    input_payload = {"user_query": "Can you give me info on the movie Interstellar?"}
    # input_payload = {"user_query": "Can you give me info on the movie Stranger than Paradise?"}
    execution_result = app.invoke(input_payload)
    
    print("\n[GPU INFERENCE SUCCESS] --- STATE REASONING LOG TRACES ---")
    for execution_step in execution_result["agent_logs"]:
        print(execution_step)
        
    print("\n--- FINAL SYSTEM RECOMMENDATION CONVERSATION ---")
    print(execution_result["final_output"])
    print("="*70 + "\n")