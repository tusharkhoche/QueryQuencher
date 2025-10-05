from langchain.text_splitter import RecursiveCharacterTextSplitter

#Split the document into chunks
def recursive_character_text_splitter(documents,size_of_chunk=1000,chunk_overlap=100 ):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=size_of_chunk, chunk_overlap=chunk_overlap,   separators=["\n\n", "\n", " ", "","."])
    docs = text_splitter.split_documents(documents)
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    return docs
