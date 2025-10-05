from rag.data_loader import load_data, get_persistent_directory
from rag.text_splitters import recursive_character_text_splitter
from rag.vector_database_creator import create_vector_database
from rag.vector_database_creator import load_vector_database
from rag.models import get_ollama_model
from rag.models import get_llama_cpp_model
from rag.prompts import get_contextualize_q_system_prompt
import torch
from rag.retrievers import get_history_aware_retriever
from rag.chains import get_question_answer_chain
from rag.prompts import get_question_answer_prompt
from rag.chains import get_rag_chain
from rag.rag import continual_chat

split_docs = None
vector_db = None


def run_program():
    torch.mps.set_per_process_memory_fraction(0.0)
    docs_with_metadata = load_data()

    persist_dir = get_persistent_directory()

    if isinstance(docs_with_metadata, list):
        global split_docs
        global vector_db

        split_docs = recursive_character_text_splitter(docs_with_metadata)
        vector_db = create_vector_database(split_docs, persist_dir)

    else:
        vector_db = load_vector_database(persist_dir)


    #llm_model = get_llama_cpp_model()
    llm_model = get_ollama_model()


    contextualize_q_system_prompt = get_contextualize_q_system_prompt()
    history_aware_retriever = get_history_aware_retriever(llm_model, persist_dir, contextualize_q_system_prompt)
    question_answer_prompt = get_question_answer_prompt()
    question_answer_chain = get_question_answer_chain(llm_model, question_answer_prompt)
    rag_chain = get_rag_chain(history_aware_retriever, question_answer_chain)
    continual_chat(rag_chain)
