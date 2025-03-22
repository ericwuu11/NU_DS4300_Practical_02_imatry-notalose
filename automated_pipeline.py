import os
import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from Preprocess_Embed_and_VectorDB import clean_text, chunk_text, extract_text_from_files, get_memory_usage
from Systems_and_LLM import get_embedding, store_embedding, retrieve_similar_documents, query_local_llm

# Defining all variations
chunk_sizes = [200, 500, 1000]
overlap_sizes = [0, 50, 100]
embedding_models = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet": "sentence-transformers/all-mpnet-base-v2",
    "GTE-Small": "thenlper/gte-small"
}
llm_models = ["llama3.2:1b", "mistral:v0.1"]
file_paths = [
    "C:/Users/jenny/Downloads/DS4300_25/Practical02/Corpus/02 - Foundations.pptx",
    "C:/Users/jenny/Downloads/DS4300_25/Practical02/Corpus/03 - Moving Beyond the Relational Model.pptx",
    "C:/Users/jenny/Downloads/DS4300_25/Practical02/Corpus/04 - Data Replication.pptx",
    "C:/Users/jenny/Downloads/DS4300_25/Practical02/Corpus/05 - NoSQL Intro + KV DBs.pptx",
    "C:/Users/jenny/Downloads/DS4300_25/Practical02/Corpus/ICS 46 Spring 2022, Notes and Examples_ AVL Trees.pdf"
]

# Sample user questions to query (input your questions here!)
user_questions = [
    "What is the purpose of data replication?",
    "Explain key-value databases with examples.",
    "How does the AVL tree maintain balance?"
]

# Output path to store results
log_dir = "./pipeline_logs"
os.makedirs(log_dir, exist_ok=True)

# Initiating full pipeline execution
cleaned_text = extract_text_from_files(file_paths)

for chunk_size in chunk_sizes:
    for overlap in overlap_sizes:
        chunk_key = f"Chunk{chunk_size}_Overlap{overlap}"
        print(f"\n>> Processing: {chunk_key}")
        chunks = chunk_text(cleaned_text, chunk_size, overlap)

        for embed_name, embed_model in embedding_models.items():
            print(f"  -> Embedding with {embed_name}")
            model = SentenceTransformer(embed_model)

            start_time = time.time()
            start_mem = get_memory_usage()
            embeddings = model.encode(chunks, convert_to_numpy=True)
            end_time = time.time()
            end_mem = get_memory_usage()

            # Storing in Redis for LLM access
            for i, (text, vec) in enumerate(zip(chunks, embeddings)):
                store_embedding(f"{chunk_key}_{embed_name}_{i}", text, vec.tolist())

            # Asking LLMs
            for question in user_questions:
                retrieved_docs = retrieve_similar_documents(question)

                for llm in llm_models:
                    response = query_local_llm(llm, question, retrieved_docs)

                    # Saving output
                    filename = f"{chunk_key}_{embed_name}_{llm.replace(':', '')}.txt"
                    filepath = os.path.join(log_dir, filename)
                    with open(filepath, "a", encoding="utf-8") as f:
                        f.write(f"Question: {question}\n")
                        f.write(f"Response:\n{response['content']}\n")
                        f.write("=" * 50 + "\n")

            print(f"    >> Time: {round(end_time - start_time, 2)}s | Mem: {round(end_mem - start_mem, 2)}MB")

print("\nAll pipelines executed and responses saved!")