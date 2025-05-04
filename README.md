RAG-Enhanced Framework for Dynamic Query Resolution in Infrastructure Inspection
Project Overview  
This project introduces a Retrieval-Augmented Generation (RAG) framework enhanced by the Mistral-7B-v0.1 Large Language Model (LLM) to improve safety, efficiency, and compliance in infrastructure inspection. The system leverages real-time data retrieval and automated report generation to provide inspectors with up-to-date regulatory standards and project-specific information.  
Key Features:
- Dynamic query resolution using Mistral-7B and PostgreSQL with PGVector.  
- Automated extraction and processing of construction documents (e.g., building codes, inspection checklists).  
- Comparative evaluation against ChatGPT, Blackbox.AI, and Gemini.AI using Gini Coefficient metrics.  
Usage Instructions  
Prerequisites  
1. Software & Tools:
   - Python 3.8+  
   - [LangChain](https://python.langchain.com/) (for query processing)  
   - PostgreSQL with [PGVector](https://github.com/pgvector/pgvector) (for vector storage)  
   - AWS Bedrock (for Mistral-7B deployment)  
   - Flask (for API endpoints)  
2. APIs & Services:
   - Mistral API (for embeddings)  
   - AWS IAM credentials with Bedrock access  
Setup Steps  
1. Data Preparation  
- PDF Extraction: Use `PyPDFLoader` from LangChain to extract text from regulatory documents, checklists, and building codes.  
  ```python
  from langchain_community.document_loaders import PyPDFLoader
  loader = PyPDFLoader("path/to/document.pdf")
  pages = loader.load()
Text Chunking: Split documents into manageable chunks using recursive splitting (aligned with Mistral-7B’s 8,192-token context window).
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = splitter.split_documents(pages)
Embedding Generation: Convert chunks into embeddings using Titan Text Embeddings v2 (via AWS Bedrock).
from langchain.embeddings import BedrockEmbeddings
embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2")
2. Database Configuration
Store embeddings in PostgreSQL with PGVector:
CREATE TABLE document_embeddings (
  id SERIAL PRIMARY KEY,
  content TEXT,
  embedding VECTOR(1536)  -- Adjust dimension based on Titan v2
);
3. Query Pipeline
Real-Time QA: Use Mistral-7B to process user queries and retrieve context from the vector database.
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=MistralLLM(),  # Configured via AWS Bedrock
    retriever=vector_db.as_retriever(),
    chain_type="stuff"
)
response = qa_chain.run("What OSHA subpart covers blasting regulations?")
4. Deployment
Containerize components using Docker:
dockerfile
# Flask API Container
FROM python:3.8
COPY requirements.txt .
RUN pip install -r requirements.txt
CMD ["flask", "run", "--host=0.0.0.0"]
•	Orchestrate with docker-compose.yml:
yaml
services:
  flask_app:
    build: ./flask_api
  postgres:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: yourpassword
Evaluation Metrics
The system’s performance is evaluated using:
1.	Gini Coefficient (0–1 scale) to measure score inequality across models (see Table 1).
2.	Human Evaluation Rubric (accuracy, relevance, clarity, terminology).

Future Work
•	Expand report generation capabilities (e.g., summaries, compliance insights).
•	Optimize scalability for larger document corpora.
•	Integrate multimodal data (images, geospatial maps).

