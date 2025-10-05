# Query Quencher 💬

Query Quencher is a personal, conversational RAG (Retrieval-Augmented Generation) project built from the ground up. It is designed to allow users to "chat" with their own documents, quenching their thirst for answers from their private knowledge base.

This project was developed to showcase the end-to-end process of building a modern AI application, with a focus on using open-source technologies to create an accessible and low-cost solution for non-technical users like students, lawyers, and researchers.

## 🎯 Core Objective

*   **Empower Non-Technical Users:** Create a tool that allows individuals to interact with their documents (text, PDF, Word) through a simple chat interface.
*   **Leverage Open-Source:** Build a cost-effective solution by primarily using open-source models, libraries, and databases.
*   **Demonstrate RAG Architecture:** Serve as a practical example of implementing a complete RAG pipeline from data ingestion to conversational response.

## ✨ Features

*   **Conversational Chat:** Engage in a continuous dialogue with your documents, with the model remembering previous parts of the conversation.
*   **Multi-Format Data Ingestion:** Supports loading and processing of `.txt`, `.pdf`, and `.docx` files.
*   **Open-Source at the Core:** Utilizes open-source LLMs (via Ollama), embedding models, and a local vector database (ChromaDB).
*   **Modular & Extensible Code:** The project is structured logically into modules for data loading, text splitting, retrieval, and chaining, making it easy to understand and extend.

## 🛠️ Tech Stack & Architecture

The RAG pipeline is orchestrated using the **LangChain** framework.

*   **Framework:** LangChain
*   **LLM:** Meta Llama 3.1 (8B) via Ollama (easily swappable)
*   **Embedding Model:** `sentence-transformers/gtr-t5-large` from Hugging Face
*   **Vector Database:** ChromaDB
*   **Core Libraries:** `langchain`, `torch`, `chroma-db`, `huggingface-hub`, `ragas`
*   **Data Loaders:** `PyPDFLoader`, `Docx2txtLoader`, `TextLoader`

### Workflow

1.  **Data Ingestion:** Local files (`.txt`, `.pdf`, `.docx`) are loaded from the `data/` directory.
2.  **Text Splitting:** Documents are split into smaller, semantically coherent chunks using a `RecursiveCharacterTextSplitter`.
3.  **Embedding & Storage:** The text chunks are converted into vector embeddings and stored locally in a **ChromaDB** vector store.
4.  **History-Aware Retrieval:** When a user asks a question, a `create_history_aware_retriever` first reformulates the query based on chat history to make it a standalone question. This query is then used to retrieve the most relevant document chunks from ChromaDB.
5.  **Augmented Generation:** The retrieved context and the user's question are passed to the LLM, which generates a factually-grounded answer.
6.  **Conversational Loop:** The process repeats, with the chat history being updated to maintain context for the next turn.

## 🚀 Getting Started

### Prerequisites

*   Python 3.8+
*   An instance of [Ollama](https://ollama.com/) running with the `llama3.1:8b` model pulled.

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/QueryQuencher.git
    cd QueryQuencher
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Add your documents:**
    Place your `.txt`, `.pdf`, and `.docx` files into the corresponding subdirectories within the `data/` folder (`data/text_files`, `data/pdf_files`, etc.).

4.  **Run the application:**
    The first time you run the application, it will process your documents, create embeddings, and build the vector database. Subsequent runs will load the existing database.
    ```bash
    python runner.py
    ```

5.  **Start chatting!**
    Follow the prompts in your terminal to start asking questions about your documents. Type `exit` to end the conversation.

## 🔮 Future Enhancements

*   **Implement Semantic Splitting:** Move from `RecursiveCharacterTextSplitter` to a semantic-based splitter for more contextually aware document chunking.
*   **Add a Re-ranker:** Introduce a re-ranking step after retrieval to improve the relevance of documents passed to the LLM.
*   **Integrate an Agent:** Develop a LangChain agent that can intelligently decide when to use the RAG chain or other tools.
*   **UI Implementation:** Build a simple web interface using Streamlit or Flask to make the application more user-friendly.

---
*Disclaimer: This is a personal project created to demonstrate skills in building RAG applications and is not intended for production use without further refinement.*
