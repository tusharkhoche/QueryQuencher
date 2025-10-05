from datasets import Dataset
import os
from dotenv import load_dotenv
from ragas import evaluate
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.chat_models import huggingface
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, ContextRecall, ContextPrecision,AnswerRelevancy

LINE_SEPARATOR = "======================"

def evaluate_rag(input,response,context):
    load_dotenv()

    evaluator_llm = LangchainLLMWrapper(ChatOllama(model="llama3.1:8b"))
    embeddings_model = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL"))

    data = {
        "user_input": input,
        "response": response,
        "retrieved_contexts": context,
        "reference": context
    }

    # Convert the data to a Hugging Face Dataset
    data = Dataset.from_dict(data)

    # Define the metrics you want to evaluate
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
    ]

    # Evaluate the dataset using the selected metrics
    results = evaluate(dataset=data,metrics=metrics,embeddings=embeddings_model,show_progress=True,llm=evaluator_llm)


    # Display the results
    df = results.to_pandas()
    print(LINE_SEPARATOR)
    print("RAG Evaluation Results")
    print(LINE_SEPARATOR)
    print(df.head())


def evaluate_test():
    load_dotenv()
    evaluator_llm = LangchainLLMWrapper(langchain_llm = ChatOllama(model="llama3.1:8b", temperature=0))

    # Example data
    data = {
        "user_input": ["What is the capital of France?"],
        "response": ["Paris is the capital of France."],
        "retrieved_contexts": [["Paris is the capital of France. It is a major European city known for its culture."]],
        "reference": ["Paris is the capital of France. It is a major European city known for its art."]
    }

    # Convert the data to a Hugging Face Dataset
    data = Dataset.from_dict(data)

    # Define the metrics you want to evaluate
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
    ]



    embeddings_model = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL"))


    #Evaluate the dataset using the selected metrics
    results = evaluate(dataset=data,metrics=metrics,embeddings=embeddings_model,show_progress=True,llm=evaluator_llm)

    # Display the results
    df = results.to_pandas()
    print(LINE_SEPARATOR)
    print("Sample Results")
    print(LINE_SEPARATOR)
    print(df.head())
