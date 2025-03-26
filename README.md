# Practical 02
This project involves building a local Retrieval-Augmented Generation (RAG) system that enables users to query a collection of course notes and receive AI-generated responses based on relevant retrieved context. The project explores different strategies for data ingestion, text processing, embedding models, vector databases, and local LLMs to analyze their impact on system performance.

# Key Features
- **Document Ingestion & Indexing:** Collect and process course notes from team members.
- **Vector Database:** Store indexed embeddings in a vector database.
- **Query Processing:** Accept user queries and retrieve relevant context.
- **LLM Response Generation:** Pass retrieved-context into a locally running LLM for response generation.
- **Performance Analysis:** Compare different configurations and evaluate retrieval accuracy, speed, and efficiency.

# Execution
Install the "Process_Embed_and_VectorDB.py," "Systems_and_LLM.py," and "automated_pipeline.py" files and store them in a folder. Within the same folder, create another folder called "Corpus." Download from our "Corpus" folder the five notes:
- "02 - Foundations"
- "03 - Moving Beyond the Relational Model"
- "04 - Data Replication"
- "05 - NoSQL Intro + KV DBs"
- "ICS 46 Spring 2022, Notes and Examples_ AVL Trees"

Run both the "automated_pipeline.py" file, and ensure that you change the file paths to where your five notes are being stored. Following this adjust the user questions to whatever you wish, then proceed to run the code. You should get a new folder called "pipeline_logs" where all the answers to your questions are stored. In the terminal you should see a list of all the processing times and memory used to run the variations of chunking strategies, embedding models, choice of Vector DB, and choice of LLMs. This process will take some time so be patient! The "Process_Embed_and_VectorDB.py" and "Systems_and_LLM.py" files are needed for the "automated_pipeline.py" to run, but are also a testing grounds of sorts if you want to run a few trials or tests, but do not want to run the whole pipeline. Thanks and happy testing!
