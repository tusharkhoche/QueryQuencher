from langchain_core.messages import AIMessage, HumanMessage
from rag.evaluation import evaluate_rag


# Function to simulate a continual chat
def continual_chat(rag_chain):
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = list()  # Collect chat history here (a sequence of messages)

    while True:

        query = input("You: ")
        if query.lower() == "exit":
            break
        else:

            # Process the user's query through the retrieval chain
            chat_history.append(HumanMessage(content=query))
            result =  rag_chain.invoke({"input": query, "chat_history": chat_history})
            print("result===>", result)
         # Display the AI's response & Update the chat history
            print("AI: ", result.get('answer'))


            chat_history.append(AIMessage(content=result.get('answer')))
            #evaluate_rag(query,result.get('answer'),result.get('context'))

