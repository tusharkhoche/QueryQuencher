
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader


BASE_DIR = os.path.dirname(os.path.abspath("data"))
DOCUMENTS_WITH_METADATA= list()

# Define the directory containing the text files and the persistent directory
def get_text_data_directory():

    data_dir = os.path.join(BASE_DIR, "data")
    text_files_dir = os.path.join(data_dir, "text_files")
    return text_files_dir


# Define the directory containing the pdf files and the persistent directory
def get_pdf_data_directory():
    data_directory = os.path.join(BASE_DIR, "data")
    pdf_files_dir = os.path.join(data_directory, "pdf_files")
    return pdf_files_dir


# Define the directory containing the MS word files and the persistent directory
def get_word_data_directory():
    data_directory = os.path.join(BASE_DIR, "data")
    word_files_dir = os.path.join(data_directory, "word_files")
    return word_files_dir

# Check if the given directory exists
def check_directory_exists(directory_name):
    if not os.path.exists(directory_name):
        raise FileNotFoundError(
            f"The directory {directory_name} does not exist. Please check the path."
        )

# Return a list of files that exists in the given directory
def list_files_in_directory(directory_name):
    files = list()
    for book in os.listdir(directory_name):
        files.append(book)
    return files

# Text file loader
def text_file_loader(file_path,file):
    loader = TextLoader(file_path)
    book_docs = loader.load()
    # Add metadata to each document indicating its source
    book_docs[0].metadata = {"source": file}
    return book_docs[0]

# Pdf file loader
def pdf_file_loader(file_path,file):
    loader = PyPDFLoader(file_path)
    book_docs = loader.load()
    # Add metadata to each document indicating its source
    book_docs[0].metadata = {"source": file}
    return book_docs[0]

# Word file loader
def word_file_loader(file_path,file):
    loader = Docx2txtLoader(file_path)
    book_docs = loader.load()
    # Add metadata to each document indicating its source
    book_docs[0].metadata = {"source": file}
    return book_docs[0]


def read_file_content_with_metadata(directory_name,files_list,file_type):

    for file in files_list:
        file_path = os.path.join(directory_name, file)

        if file_type.lower() == "txt":
            doc = text_file_loader(file_path,file)
            DOCUMENTS_WITH_METADATA.append(doc)
            print("Total No Of Text Files-", len(files_list))

        if file_type.lower() == "pdf":
            doc = pdf_file_loader(file_path, file)
            DOCUMENTS_WITH_METADATA.append(doc)
            print("Total No Of PDF Files-", len(files_list))

        if file_type.lower() == "word":
            doc = word_file_loader(file_path, file)
            DOCUMENTS_WITH_METADATA.append(doc)
            print("Total No Of MS Word Files-", len(files_list))





def load_data():
    persist_directory = get_persistent_directory()
    # Check if the Chroma vector store already exists
    if not os.path.exists(persist_directory):
        print("Persistent directory does not exist. Initializing vector store...")

        # Ensure the data directory exists
        check_directory_exists(get_text_data_directory())
        check_directory_exists(get_pdf_data_directory())
        check_directory_exists(get_word_data_directory())

        # List all text files in the directory
        text_files_list = list_files_in_directory(get_text_data_directory())

        # List all pdf files in the directory
        pdf_files_list  = list_files_in_directory(get_pdf_data_directory())

        # List all word files in the directory
        word_files_list  = list_files_in_directory(get_word_data_directory())

        # Read the text content from each file and store it with metadata
        read_file_content_with_metadata(directory_name = get_text_data_directory(),files_list=text_files_list,file_type="txt")
        read_file_content_with_metadata(directory_name = get_pdf_data_directory(), files_list=pdf_files_list,file_type="pdf")
        read_file_content_with_metadata(directory_name = get_word_data_directory(),files_list=word_files_list, file_type="word")
        return DOCUMENTS_WITH_METADATA
    else:
        return "Vector store already exists. No need to initialize."


def get_persistent_directory():
    persistent_directory = os.path.join(BASE_DIR, "vector_database")
    return persistent_directory




