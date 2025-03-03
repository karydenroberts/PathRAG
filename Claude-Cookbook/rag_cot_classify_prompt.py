import vectordb
import textwrap
import re
import os
import anthropic
from verification_module import verify_classification

# Initialize Anthropic client
try:
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
except Exception as e:
    print(f"Error initializing Anthropic client: {e}")
    raise

def rag_chain_of_thought_classify(X, verify_result=False):
    """
    Classify a customer support ticket using RAG and chain-of-thought reasoning.
    
    Parameters:
    - X: The customer support query text
    - verify_result: If True, verifies the classification using verification_module
    
    Returns:
    - The classification result
    """
    # Get similar examples from vector database
    try:
        rag = vectordb.search(X, 5)
        rag_string = ""
        for example in rag:
            rag_string += textwrap.dedent(f"""
            <example>
                <query>
                    "{example["metadata"]["text"]}"
                </query>
                <label>
                    {example["metadata"]["label"]}
                </label>
            </example>
            """)
    except Exception as e:
        print(f"Warning: Vector search failed: {e}")
        rag_string = ""
    
    # Format the prompt
    prompt = textwrap.dedent("""
    You will classify a customer support ticket into one of the following categories:
    <categories>
        {{categories}}
    </categories>

    Here is the customer support ticket:
    <ticket>
        {{ticket}}
    </ticket>

    Use the following examples to help you classify the query:
    <examples>
        {{examples}}
    </examples>

    First you will think step-by-step about the problem in scratchpad tags.
    You should consider all the information provided and create a concrete argument for your classification.
    
    Respond using this format:
    <response>
        <scratchpad>Your thoughts and analysis go here</scratchpad>
        <category>The category label you chose goes here</category>
    </response>
    """).replace("{{categories}}", categories).replace("{{ticket}}", X).replace("{{examples}}", rag_string)
    
    # Generate classification with Claude
    try:
        response = client.messages.create( 
            messages=[{"role":"user", "content": prompt}, {"role":"assistant", "content": "<response><scratchpad>"}],
            stop_sequences=["</category>"], 
            max_tokens=4096, 
            temperature=0.0,
            model="claude-3-haiku-20240307"
        )
        
        # Extract the scratchpad and category from the response
        full_response = response.content[0].text + "</category>"
        
        # Use regex to extract scratchpad content and category
        scratchpad_match = re.search(r'<scratchpad>(.*?)</scratchpad>', full_response, re.DOTALL)
        category_match = re.search(r'<category>(.*?)</category>', full_response, re.DOTALL)
        
        scratchpad = scratchpad_match.group(1).strip() if scratchpad_match else ""
        result = category_match.group(1).strip() if category_match else ""
        
        # Verify the classification if requested
        if verify_result and result:
            is_verified = verify_classification(X, scratchpad, result)
            if not is_verified:
                # Could add fallback logic here if verification fails
                print(f"Warning: Classification '{result}' failed verification")
        
        return result
    except Exception as e:
        print(f"Error during classification: {e}")
        return "Error: Classification failed"