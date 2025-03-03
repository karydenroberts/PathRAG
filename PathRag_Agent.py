import asyncio
from PathRAG.PathRAG import PathRAG
from openai import AsyncOpenAI

# Initialize the PathRAG system
pathrag = PathRAG(
    working_dir="./pathrag_code_knowledge",
    graph_storage="NetworkXStorage",  # Default, can be changed to Neo4J later
    embedding_func=your_embedding_function  # Use the same embeddings across systems
)

# Create an agent that uses PathRAG
async def coding_knowledge_agent(query):
    # Use PathRAG to retrieve relevant context
    from PathRAG.base import QueryParam
    
    param = QueryParam(
        mode="hybrid",
        top_k=40,  # Number of nodes to retrieve
        response_type="Detailed"
    )
    
    # Get the response from PathRAG
    pathrag_response = await pathrag.aquery(query, param)
    
    # You can use the response directly or enhance it with a follow-up LLM call
    return pathrag_response

# Example usage
async def main():
    # First populate the knowledge base
    with open("some_codebase_file.py", "r") as f:
        code_content = f.read()
    
    # Insert code into PathRAG
    await pathrag.ainsert(code_content)
    
    # Query the system
    response = await coding_knowledge_agent("How does the error handling work?")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())