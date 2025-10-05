from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever

def get_database_retriever(persistent_directory):
    load_dotenv()

    embeddings_model = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL"),multi_process = True)

    # Load the existing vector store with the embedding function
    chroma_db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings_model)

    # Create a retriever for querying the vector store
    # `search_type` specifies the type of search (e.g., similarity)
    # `search_kwargs` contains additional arguments for the search (e.g., number of results to return)
    retriever = chroma_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )
    return retriever

# It helps LLM to reformulate the question based on chat history
def get_history_aware_retriever(model,persistent_directory, contextualize_q_system_prompt):
    retriever = get_database_retriever(persistent_directory)

    history_aware_retriever = create_history_aware_retriever(
        model, retriever, contextualize_q_system_prompt
    )
    return history_aware_retriever



