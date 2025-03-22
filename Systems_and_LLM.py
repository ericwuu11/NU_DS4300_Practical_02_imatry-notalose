import ollama
import redis
import numpy as np
from redis.commands.search.query import Query

# Initializing Redis connection
redis_client = redis.Redis(host="localhost", port=6379, db=0)

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

SYSTEM_PROMPT = """
You are an advanced AI system designed to retrieve relevant information from a Redis vector database.
Your goal is to provide the most accurate and contextually relevant answers to user queries.
You will:
1. Convert user queries into embeddings using efficient text models.
2. Use vector search in Redis with HNSW indexing to retrieve the closest matches.
3. Rank results based on similarity scores and present the top responses.
4. Provide clear and concise explanations in natural language.
"""

# Generating an embedding using Ollama
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

# Storing embedding in Redis
def store_embedding(doc_id: str, text: str, embedding: list):
    key = f"{DOC_PREFIX}{doc_id}"
    redis_client.hset(
        key,
        mapping={
            "text": text,
            "embedding": np.array(embedding, dtype=np.float32).tobytes(),
        },
    )

# Function to retrieve most similar documents from Redis
def retrieve_similar_documents(query_text):
    query_embedding = get_embedding(query_text)

    q = (
        Query("*=>[KNN 3 @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("text", "vector_distance")
        .dialect(2)
    )

    res = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": np.array(query_embedding, dtype=np.float32).tobytes()}
    )

    retrieved_docs = [doc.text for doc in res.docs]
    return retrieved_docs

# Function to query a local LLM
def query_local_llm(model_name, query_text, retrieved_docs):
    context = "\n".join(retrieved_docs)
    formatted_query = f"Context:\n{context}\n\nUser Question: {query_text}\n\nAnswer:"

    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": formatted_query}]
    )
    return response["message"]

# Main loop for user interaction
if __name__ == "__main__":
    # Storing documents in Redis
    texts = [
        "Redis is an in-memory key-value database.",
        "Ollama provides efficient LLM inference on local machines.",
        "Vector databases store high-dimensional embeddings for similarity search.",
        "HNSW indexing enables fast vector search in Redis.",
        "Ollama can generate embeddings for RAG applications.",
    ]

    for i, text in enumerate(texts):
        embedding = get_embedding(text)
        store_embedding(str(i), text, embedding)

    print("\nWelcome! You can ask questions based on stored knowledge.")
    print("Type 'exit' to stop.")

    while True:
        # Get user input
        query_text = input("\nEnter your question: ")
        if query_text.lower() == "exit":
            print("Goodbye!")
            break

        # Retrieving relevant documents
        retrieved_docs = retrieve_similar_documents(query_text)

        print("\nTop Retrieved Documents:")
        for doc in retrieved_docs:
            print(f"- {doc}")

        # Getting Llama 2 response
        print("\nLlama 3 Response:")
        print(query_local_llm("llama3.2:1b", query_text, retrieved_docs))

        # Getting Mistral response
        print("\nMistral Response:")
        print(query_local_llm("mistral:v0.1", query_text, retrieved_docs))