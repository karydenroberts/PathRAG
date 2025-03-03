# PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths

This is the official implementation of **PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths**.

## Overview

PathRAG is a novel approach to graph-based Retrieval Augmented Generation (RAG) that addresses two key limitations of existing methods:
1. **Information redundancy** in retrieved content
2. **Suboptimal flat prompting structures** that fail to prioritize relevant information

PathRAG solves these problems through:
- **Path-centric retrieval**: Identifies and retrieves key relational paths between query-relevant nodes
- **Flow-based pruning**: Uses resource allocation with decay rate (α) to reduce redundancy
- **Structured prompting**: Orders paths by reliability to enhance LLM response quality

## Key Features

- **Three-stage pipeline**: Node Retrieval → Path Retrieval → Answer Generation
- **Flow-based path pruning**: Reduces noise while preserving critical information
- **Path reliability scoring**: Ranks paths by average resource flow
- **Token efficiency**: Reduces token consumption by 16-44% compared to baselines
- **Performance**: Achieves 60.44% average win rate against GraphRAG across six datasets

## Architecture

PathRAG's architecture consists of:
1. **Node Retrieval**: Extracts keywords from queries and uses dense vector matching to retrieve nodes
2. **Path Retrieval**: 
   - Identifies paths between node pairs
   - Applies flow-based pruning with parameters α (decay rate) and θ (early stopping)
   - Selects top-K paths based on reliability scores
3. **Answer Generation**: 
   - Orders paths by ascending reliability to mitigate "lost in the middle" effects
   - Generates comprehensive responses with structured knowledge

## Experimental Results

PathRAG outperforms state-of-the-art baselines across six datasets:
- Agriculture, Legal, History, CS, Biology, and Mix
- Superior performance on five evaluation dimensions: Comprehensiveness, Diversity, Logicality, Relevance, and Coherence
- Larger datasets show more significant improvements (e.g., 65.53% win rate on Legal dataset)

## Implementation Details

- Uses GPT-4o-mini as the LLM for response generation
- Optimal parameters: N=40 (retrieved nodes), K=15 (paths), α=0.8 (decay rate)
- PathRAG uses 13,318 tokens per query (16% reduction vs. LightRAG)
- PathRAG-It variant achieves 44% token reduction with comparable performance

## File By File Analysis

# PathRAG Code Analysis

## 1. PathRAG.py
The core class implementing the PathRAG system. This file handles:
- Document insertion and chunking
- Knowledge graph construction
- Path-based retrieval and query handling
- Storage initialization and configuration
- Interface to LLM models

## 2. utils.py
Utility functions supporting the overall system:
- Data handling with JSON load/save functions
- Text processing and token management
- Embedding functions with quantization
- Caching mechanisms
- Async function limiting and semaphores

## 3. storage.py
Implements storage backends:
- `JsonKVStorage`: Key-value storage using JSON files
- `NanoVectorDBStorage`: Vector database using NanoVectorDB
- `NetworkXStorage`: Graph storage using NetworkX
- Methods for entity and relationship management

## 4. operate.py
Contains core operational algorithms:
- Text chunking mechanisms
- Entity and relationship extraction
- Path finding and retrieval algorithms
- Query processing and context building
- Flow-based path pruning implementation

## 5. llm.py
LLM provider interfaces:
- OpenAI, Azure, NVIDIA API integrations
- Embedding functions for different providers
- Retry mechanisms and error handling
- Streaming response support

## 6. prompt.py
Prompt templates and constants:
- Entity extraction prompts
- RAG response formatting templates
- Keyword extraction prompts
- Example formats for zero-shot learning

## 7. base.py
Base classes and interfaces:
- `StorageNameSpace`: Base class for storage components
- `BaseVectorStorage`, `BaseKVStorage`, `BaseGraphStorage`: Storage interfaces
- `QueryParam`: Query configuration parameters
- Type definitions for the system

## Citation

If you use this code, please cite the paper:
```
@article{PathRAG,
  title={PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths},
  author={Chen, Boyu and Guo, Zirui and Yang, Zidan and Chen, Yuluo and Chen, Junze and Liu, Zhenghao and Shi, Chuan and Yang, Cheng},
  journal={},
  year={}
}
```

## Contact

For any questions or issues, please contact the authors from Beijing University of Posts and Telecommunications.