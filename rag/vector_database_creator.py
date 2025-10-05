import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv


def create_vector_database(splitted_docs,persistent_directory):
    load_dotenv()
    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings_model = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL"),multi_process = True)
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it
    print("\n--- Creating and persisting vector store ---")

    vector_database = Chroma.from_documents(splitted_docs,embeddings_model,persist_directory=persistent_directory)
    return vector_database


def load_vector_database(persistent_directory):
    load_dotenv()
    embeddings_model = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL"),multi_process = True)
    # Create the vector store and persist it
    print("\n--- Persisting vector store ---")
    vector_database = Chroma(persist_directory=persistent_directory, embedding_function=embeddings_model)
    return vector_database