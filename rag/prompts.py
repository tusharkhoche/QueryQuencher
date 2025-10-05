from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


    # Contextualize question prompt
    # This system prompt helps the AI understand that it should reformulate the question
    # based on the chat history to make it a standalone question
def get_contextualize_q_system_prompt():
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history in a single string strictly. Do NOT answer the question, just "
        "reformulate it in a single string strictly if needed otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    return contextualize_q_prompt

def get_question_answer_prompt():
    # Answer question prompt
    # Instructing The Model To Avoid Adding False Information
    # This system prompt helps the AI understand that it should provide concise answers
    # based on the retrieved context and indicates what to do if the answer is unknown
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of retrieved context and chat history to answer the question."
        "If the answer is not present in the pieces of retrieved context or in the chat history, say that you don't know." 
        "Use five sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )

    # Create a prompt template for answering questions
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    return qa_prompt




