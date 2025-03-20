# Practical 02
This project involves building a local Retrieval-Augmented Generation (RAG) system that enables users to query a collection of course notes and receive AI-generated responses based on relevant retrieved context. The project explores different strategies for data ingestion, text processing, embedding models, vector databases, and local LLMs to analyze their impact on system performance.

# Key Features
- **Document Ingestion & Indexing:** Collect and process course notes from team members.
- **Vector Database:** Store indexed embeddings in a vector database.
- **Query Processing:** Accept user queries and retrieve relevant context.
- **LLM Response Generation:** Pass retrieved-context into a locally running LLM for response generation.
- **Performance Analysis:** Compare different configurations and evaluate retrieval accuracy, speed, and efficiency.

# Execution
Install the Process_Embed_and_VectorDB.py and Systems_and_LLM.py files and store them in a folder. Within the same folder, create another folder called "Corpus." Download from our "Corpus" folder the five notes:
- "02 - Foundations"
- "03 - Moving Beyond the Relational Model"
- "04 - Data Replication"
- "05 - NoSQL Intro + KV DBs"
- "ICS 46 Spring 2022, Notes and Examples_ AVL Trees"

Run both .py files in PyCharm, and ask any question you would like to the Systems_and_LLM.py file. This process will take some time so give it around 10-15 minutes. Once you are done prompting the LLM your questions, type 'exit' to leave the LLM.
