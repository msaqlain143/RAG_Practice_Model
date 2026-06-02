import os
from langchain_community.llms import Ollama

# -------------------------------------------------------------
# 1. THE EXACT SAME LONG-TAIL DATASET LAYER
# -------------------------------------------------------------
# Notice that this baseline has NO ACCESS to the external validation search tool!
VECTOR_KNOWLEDGE_BASE = {
    "interstellar": {
        "title": "Interstellar",
        "year": "2014",
        "director": "Christopher Nolan",
        "genre": "Sci-Fi, Adventure, Drama",
    },
    "stranger than paradise": {
        "title": "Stranger than Paradise",
        "year": "UNKNOWN",  # The data gap that triggers hallucinations
        "director": "Jim Jarmusch",
        "genre": "Indie, Comedy-Drama",
    }
}

# Connect directly to your local Llama-3 model engine
local_brain = Ollama(model="llama3", temperature=0)

def run_baseline_rag(user_query: str):
    print(f"\n[USER QUERY]: {user_query}")
    
    # --- STEP 1: LINEAR RETRIEVAL ---
    query_lower = user_query.lower()
    target_key = "interstellar" if "interstellar" in query_lower else "stranger than paradise"
    
    retrieved_context = VECTOR_KNOWLEDGE_BASE.get(target_key)
    print(f"[RETRIEVAL LAYER]: Ingested raw database attributes for: '{retrieved_context['title']}'")
    
    # --- STEP 2: BLIND PROMPT CONSTRUCTION (NO AUDITING/REFORMULATION) ---
    # The sparse metadata is passed directly to the generator
    prompt = f"""
    Context: You are a Conversational Recommender System. Recommend the movie below to the user based on these attributes:
    Title: {retrieved_context['title']}
    Year: {retrieved_context['year']}
    Director: {retrieved_context['director']}
    Genre: {retrieved_context['genre']}
    
    Provide a professional conversational recommendation response.
    """
    
    print("[GENERATION LAYER]: Sending raw context directly to Llama-3 parameter paths...")
    
    # --- STEP 3: LINEAR GENERATION ---
    ai_response = local_brain.invoke(prompt)
    return ai_response

if __name__ == "__main__":
    print("\n" + "="*70)
    print("         LAUNCHING STANDARD LINEAR BASELINE RAG PIPELINE")
    print("="*70)
    
    # Test the exact same cold-start movie query
    query = "Can you give me info on the movie Stranger than Paradise?"
    final_output = run_baseline_rag(query)
    
    print("\n--- BASELINE SYSTEM RESPONSE ---")
    print(final_output)
    print("="*70 + "\n")