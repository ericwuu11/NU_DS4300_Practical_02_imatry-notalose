# Practical 02
This project involves building a local Retrieval-Augmented Generation (RAG) system that enables users to query a collection of course notes and receive AI-generated responses based on relevant retrieved context. The project explores different strategies for data ingestion, text processing, embedding models, vector databases, and local LLMs to analyze their impact on system performance.

# Key Features
- Document Ingestion & Indexing: Collect and process course notes from team members.
- Vector Database: Store indexed embeddings in a vector database (Redis, Chroma, and another of your choice).
- Query Processing: Accept user queries and retrieve relevant context.
- LLM Response Generation: Pass retrieved context into a locally running LLM for response generation.
- Performance Analysis: Compare different configurations and evaluate retrieval accuracy, speed, and efficiency.
