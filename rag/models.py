from langchain_community.llms import LlamaCpp
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

def get_ollama_model():
    return ChatOllama(model="llama3.1:8b", temperature=0)

def get_chat_groq_model():
    return ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0)

def get_llama_cpp_model():

    llm = LlamaCpp(
        model_path="llms/Meta-Llama-3.1-8B-Instruct-Q4_0_4_4.gguf",

        temperature=0,
        n_ctx=128000,
        max_tokens=8192,
        #echo=False,
        top_p=1.0,
        f16_kv=True,
        verbose=True
     )

    return llm