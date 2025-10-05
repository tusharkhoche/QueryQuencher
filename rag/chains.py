from langchain.chains import  create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# Create a chain to combine documents for question answering
# `create_stuff_documents_chain` feeds all retrieved context into the LLM
def get_question_answer_chain(model, qa_prompt):
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    return question_answer_chain


# Create a retrieval chain that combines the history-aware retriever and the question answering chain
def get_rag_chain(history_aware_retriever,question_answer_chain):

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain




