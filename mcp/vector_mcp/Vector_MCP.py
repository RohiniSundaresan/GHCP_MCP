
import mcp
from mcp.server.fastmcp import FastMCP
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import csv_loader
from langchain_community.vectorstores import Chroma
import os
from datetime import datetime

# Initialize FastMCP server
mcp = FastMCP("ChromadbOperations", timeout=120)

# Create embeddings using HuggingFaceEmbeddings
print("Loaded embeddings")

def db_connect():
    """
    Handle database connection.
    Returns:
        str: Path to the embedding directory.
    """
    embedding_path = "C:/GenAI/ghcp_mcp/data/embed_data"
    try:
        os.makedirs(embedding_path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory: {e}")
    return embedding_path

def create_vector_store_method(embedding_path: str, file_name: str):
    """
    Create a vector store using Chroma and store the embeddings in a directory.
    Args:
        embedding_path (str): Path to the embedding directory.
        file_name (str): Name of the file to be stored in the vector store.
    Returns:
        None    
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    filepath = f"C:/GenAI/ghcp_mcp/data/input/{file_name}"
    
    loader = csv_loader.CSVLoader(filepath, encoding="utf-8")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    
    vector_store = Chroma.from_documents(documents, embeddings, persist_directory=embedding_path)
    vector_store.persist()
    print("Vector store created and persisted")

def retrieve_documents_method(query: str, n_results: int, embedding_path: str):
    """
    Retrieve similar documents for the given query and store them in a file.
    Args:
        query (str): The query for which to retrieve similar documents.
        n_results (int): The number of similar documents to retrieve.
        embedding_path (str): Path to the embedding directory.
    Returns:
        list: A list of similar documents retrieved from the vector store.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    retrieval_dir = "C:/GenAI/ghcp_mcp/data/retrieval_context"
    
    try:
        n_results = int(n_results)
    except ValueError:
        print("Invalid value for n_results. Defaulting to 1.")
        n_results = 1
    
    try:
        os.makedirs(retrieval_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory: {e}")
    
    vector_store = Chroma(persist_directory=embedding_path, embedding_function=embeddings)
    retrieved_docs = vector_store.similarity_search(query, k=n_results)
    
    context = ""
    for doc in retrieved_docs:
        content = doc.page_content
        context += (content + '\n\n')
    
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_path = f"{retrieval_dir}/TestCase_Context_{timestamp}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(context)
    
    print(f"Saved retrieved documents to {output_path}")
    return retrieved_docs

@mcp.tool()
def create_vector_store(file_name: str):
    """
    MCP tool to create a vector store.
    Args:
        file_name (str): Name of the file to be stored in the vector store.
    """
    embedding_path = db_connect()
    create_vector_store_method(embedding_path, file_name)

@mcp.tool()
def retrieve_similar_documents(query: str, n_results: int):
    """
    MCP tool to retrieve similar documents.
    Args:
        query (str): The query for which to retrieve similar documents.
        n_results (int): The number of similar documents to retrieve.
    """
    embedding_path = db_connect()
    return retrieve_documents_method(query, n_results, embedding_path)

@mcp.tool()
def create_prompt_template(query: str) -> str:
    """
    Create a prompt template for the given query.
    Args:
        query (str): The query for which to create the prompt template.
    Returns:
        str: The prompt template with the query embedded.
    """
    prompt_template = f"Act as a BDD expert, understand the query: {query}, and generate the BDD feature file by strictly adhering to the Cucumber syntax."
    return prompt_template

@mcp.tool()
def store_and_retrieve(file_name: str, query: str, n_results: int):
    """
    MCP tool to create a vector store and retrieve similar documents in one method.
    Args:
        file_name (str): Name of the file to be stored in the vector store.
        query (str): The query for which to retrieve similar documents.
        n_results (int): The number of similar documents to retrieve.
    """
    embedding_path = db_connect()
    create_vector_store_method(embedding_path, file_name)
    return retrieve_documents_method(query, n_results, embedding_path)



 
# if __name__ == "__main__":
#     # create_vector_store()
#     query_text = "User should be able to enter zip, state and find a Dentist in the Metlife application"
#     # result = retrieve_similar_documents(query=query_text, n_results=1)
#     # print("Retrieved document:", result)
#     prompt = create_prompt_template(query_text)
#     print("Prompt Template:", prompt)
 