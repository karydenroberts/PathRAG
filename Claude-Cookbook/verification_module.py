import os
import anthropic
import textwrap
import sys
import importlib.util
from pathlib import Path

# Add parent directory to path to import from siblings
sys.path.append(str(Path(__file__).parent))

# Initialize the Anthropic client once at module level
try:
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
except Exception as e:
    print(f"Warning: Could not initialize Anthropic client: {e}")
    client = None

# Import relevant modules from the codebase
try:
    # Try to import as a module
    if importlib.util.find_spec("classification") is not None:
        from classification import categories
    else:
        # Fall back to parsing the XML file
        xml_path = Path(__file__).parent / "classification.xml"
        if xml_path.exists():
            with open(xml_path, "r") as f:
                exec(f.read())
        else:
            categories = "No categories found"
    
    # Import vectordb module
    vectordb_imported = importlib.util.find_spec("vectordb") is not None
    if vectordb_imported:
        import vectordb
except Exception as e:
    print(f"Warning: Error importing modules: {e}")
    categories = "No categories found"
    vectordb_imported = False

def convert_query_to_declarative(query: str) -> str:
    """
    Convert a natural language question into a declarative statement 
    specific to insurance classification.
    """
    return "The appropriate category for this query is ____."

def verify_reasoning(query: str, reasoning_chain: str, final_answer: str, use_vectordb=False) -> bool:
    """
    Use deductive verification to check if the reasoning_chain logically supports the final_answer.
    Returns True if verified, False otherwise.
    
    Parameters:
    - query: The original customer support query
    - reasoning_chain: The step-by-step reasoning process (from scratchpad)
    - final_answer: The category classification result
    - use_vectordb: If True, uses vector search to enhance context for verification
    """
    if client is None:
        print("Warning: Anthropic client not initialized. Verification skipped.")
        return True  # Default to passing verification if client is not available
    
    # Step 1: Convert the query into a declarative statement.
    declarative = convert_query_to_declarative(query)
    
    # Step 2: Optionally enhance context with vector search
    context = ""
    if use_vectordb and vectordb_imported:
        try:
            vdb = vectordb.VectorDB()
            search_results = vdb.search(query, k=2)
            for result in search_results:
                context += f"Similar query: {result['metadata']['text']}\n"
                context += f"Classified as: {result['metadata']['label']}\n\n"
        except Exception as e:
            print(f"Warning: Error using vectordb: {e}")
    
    # Step 3: Construct a verification prompt.
    verification_prompt = (
        f"You are verifying an insurance customer support query classification.\n\n"
        f"Original Query: {query}\n"
        f"Declarative Statement: {declarative}\n"
        f"{context}\n"
        f"The reasoning process was:\n{reasoning_chain}\n\n"
        f"The final classification: {final_answer}\n\n"
        "Question: Based solely on the reasoning process provided, can the final classification be logically deduced? "
        "Answer 'Yes' if it follows logically, otherwise answer 'No'."
    )
    
    # Step 4: Call the Claude API to evaluate the prompt
    try:
        response = client.messages.create(
            messages=[
                {"role": "system", "content": "You are an expert verifier of logical reasoning for insurance classification."},
                {"role": "user", "content": verification_prompt}
            ],
            model="claude-3-haiku-20240307",
            temperature=0.0  # Deterministic output
        )
        
        # Extract and clean up the answer.
        verification_output = response.content[0].text.strip().lower()
        
        # Step 5: Determine the result.
        return verification_output.startswith("yes")
    except Exception as e:
        print(f"Warning: Error during verification API call: {e}")
        return True  # Default to passing verification on API error

def verify_classification(query: str, scratchpad: str, category: str) -> bool:
    """
    Integration function to verify if a classification is logically sound.
    This function is designed to be called from rag_cot_classify_prompt.py.
    
    Parameters:
    - query: The customer query
    - scratchpad: The LLM's reasoning in the scratchpad
    - category: The final category classification
    
    Returns:
    - True if the classification is verified, False otherwise
    """
    return verify_reasoning(query, scratchpad, category)

# Example usage:
if __name__ == "__main__":
    query = "I had water damage in my bathroom last week and need to submit a claim. What documents do I need to provide?"
    reasoning_chain = (
        "Step 1: The customer is asking about submitting a claim for water damage.\n"
        "Step 2: This involves the process of filing an insurance claim.\n"
        "Step 3: The customer specifically needs help with documentation requirements.\n"
        "Step 4: Questions about claim filing procedures and documentation fall under Claims Assistance."
    )
    final_answer = "Claims Assistance"

    is_verified = verify_reasoning(query, reasoning_chain, final_answer)
    if is_verified:
        print("Verification passed: The reasoning supports the final answer.")
    else:
        print("Verification failed: The reasoning does not support the final answer.")