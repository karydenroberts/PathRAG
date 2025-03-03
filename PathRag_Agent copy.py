import asyncio
import os
from PathRAG.PathRAG import PathRAG
from PathRAG.llm import get_embedding_function
from PathRAG.base import QueryParam

# 1. Setup embedding function
embedding_func = get_embedding_function(
    embedding_model="voyage-code-3",  # Updated model version
    embedding_dimension=1024,         # Default dimension (alternatives: 256, 512, 2048)
    cache_folder="./embedding_cache"
)

# 2. Initialize PathRAG system
# Configure with specific parameters
pathrag = PathRAG(
    working_dir="./pathrag_code_knowledge",
    graph_storage="NetworkXStorage",
    embedding_func=embedding_func,
    config={
        "chunk_size": 1000,                   # Text chunk size
        "chunk_overlap": 200,                 # Overlap between chunks
        "embedding_batch_num": 32,            # Batch size for embedding
        "cosine_better_than_threshold": 0.25, # Similarity threshold
        "node2vec_params": {                  # Graph embedding parameters
            "dimensions": 128,
            "walk_length": 80,
            "num_walks": 10,
            "p": 1,
            "q": 1
        },
        "context_length": 32000               # Updated context length
    }
)
# 3. Function to populate knowledge base
async def populate_knowledge_base(directory_path):
    file_count = 0
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(('.py', '.js', '.md', '.txt')):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        await pathrag.ainsert(
                            content,
                            metadata={"filename": file, "path": file_path}
                        )
                        file_count += 1
                        print(f"Processed: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    print(f"Successfully embedded {file_count} files into PathRAG")

# 4. Query function
async def query_knowledge_base(question):
    param = QueryParam(
        mode="hybrid",     # Combines vector and graph-based retrieval
        top_k=40,          # Number of nodes to retrieve
        path_k=15,         # Number of paths to retrieve  
        alpha=0.8,         # Flow decay rate
        response_type="Detailed"
    )
    
    response = await pathrag.aquery(question, param)
    return response

# 5. Main function
async def main():
    # First populate the knowledge base
    await populate_knowledge_base("/path/to/your/codebase")
    
    # Then query it
    while True:
        question = input("\nAsk a question about the codebase (or 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        
        response = await query_knowledge_base(question)
        print("\n" + "=" * 80)
        print(response)
        print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())