import os
from datetime import datetime
from mcp.server.fastmcp import FastMCP
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
#from langchain_chroma import Chroma
from langchain_community.document_loaders import csv_loader
import pandas as pd

mcp = FastMCP("VectorStorage")

def _ensure_dir(path):
    if os.path.isfile(path):
        raise ValueError(f"Path '{path}' is a file, not a directory. Please provide a directory path.")
    os.makedirs(path, exist_ok=True)

def _excel_to_csv(excel_path):
    csv_path = os.path.splitext(excel_path)[0] + ".csv"
    df = pd.read_excel(excel_path)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path

@mcp.tool(name="vector_store", description="Create a vector store from an excel file and store embeddings.")
def vector_store(contextpath: str, embedding_path: str, model_name: str):
    _ensure_dir(embedding_path)
    if contextpath.lower().endswith((".xls", ".xlsx")):
        contextpath = _excel_to_csv(contextpath)
    loader = csv_loader.CSVLoader(file_path=contextpath, encoding="utf-8")
    documents = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vector_store = Chroma.from_documents(documents, embeddings, persist_directory=embedding_path)
    vector_store.persist()
    return f"Vector store created and persisted with {len(documents)} documents."

@mcp.tool(name="retrieve_context", description="Retrieve similar documents for a query and save to file.")
def retrieve_context(query: str, persist_directory: str, retrieval_directory: str, model_name: str, n_results: int = 1):
    _ensure_dir(retrieval_directory)
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retrieved_docs = vector_store.similarity_search(query, k=int(n_results))
    context_str = "\n\n".join(doc.page_content for doc in retrieved_docs)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_path = os.path.join(retrieval_directory, f"TestCase_Context_{timestamp}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(context_str)
    #return f"Saved {len(retrieved_docs)} retrieved documents to {output_path}"
    return context_str

# write a function to read the user stories from excel file, 
# create a string by concatenating user story description and 
# acceptance criteria and send it retrieve the context

@mcp.tool(name="generate_prompt_for_BDD", description="Generate BDD Prompt for each each user story in an excel file.")
def generate_prompt_for_BDD(input_excel_path: str, persist_directory: str, 
        retrieval_directory: str, model_name: str, n_results: int = 1):
    _ensure_dir(retrieval_directory)
    # Read and preprocess the Excel file: drop rows with missing user stories or acceptance criteria, strip whitespace
    df = pd.read_excel(input_excel_path)
    df = df.dropna(subset=["Description", "AcceptanceCriteria"])
    df["Description"] = df["Description"].astype(str).str.strip()
    df["AcceptanceCriteria"] = df["AcceptanceCriteria"].astype(str).str.strip()
    for idx, row in df.iterrows():
        query = f"{row['Description']} {row['FunctionalRequirements']} {row['AcceptanceCriteria']} {row['Assumptions']}"
        context_str = retrieve_context(
            query=query,
            persist_directory=persist_directory,
            retrieval_directory=retrieval_directory,
            model_name=model_name,
            n_results=n_results
        )
        # generate prompt
        output = get_prompt_template(query, context_str)
        # Ensure 'Prompts' directory exists inside retrieval_directory
        prompts_dir = os.path.join(retrieval_directory, "Prompts")
        _ensure_dir(prompts_dir)
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        user_story_id = row.get("UserStoryID", idx + 1)
        output_filename = f"UserStory_{user_story_id}_{timestamp}.txt"
        output_path = os.path.join(prompts_dir, output_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output)
    
        print(f"Generated BDD for User Story {idx + 1}:")
        print(output)
        print("-" * 80)


@mcp.prompt(name="get_us_to_bdd_prompt", description="Get BDD prompt from user story and context.")
def get_prompt_template(user_story: str, context: str) -> str:
    template_path = os.path.join(os.path.dirname(__file__), '../../resources/prompt_template.txt')
    with open(template_path, 'r', encoding='utf-8') as f:
        prompt_str = f.read()
    return prompt_str.format(user_story=user_story, context=context)

# def generate_prompt_for_playwright(input_excel_path: str,retrieval_directory: str):
#     # Read and preprocess the Excel file: drop rows with missing user stories or acceptance criteria, strip whitespace
#     df = pd.read_excel(input_excel_path)
#     df = df.dropna(subset=["Description", "AcceptanceCriteria", "FeatureFile"])
#     df["Description"] = df["Description"].astype(str).str.strip()
#     df["AcceptanceCriteria"] = df["AcceptanceCriteria"].astype(str).str.strip()
#     df["FeatureFile"] = df["FeatureFile"].astype(str).str.strip()
#     for idx, row in df.iterrows():
#         query = f"{row['Description']} {row['AcceptanceCriteria']} {row['FeatureFile']}"
        
#         # generate prompt
#         output = get_prompt_template_playwright_stepdef(query)
#         # Ensure 'Prompts' directory exists inside retrieval_directory
#         prompts_dir = os.path.join(retrieval_directory, "Prompts")
#         _ensure_dir(prompts_dir)
#         timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
#         user_story_id = row.get("UserStoryID", idx + 1)
#         output_filename = f"UserStory_{user_story_id}_{timestamp}.txt"
#         output_path = os.path.join(prompts_dir, output_filename)
#         with open(output_path, "w", encoding="utf-8") as f:
#             f.write(output)
    
#         print(f"Generated BDD for User Story {idx + 1}:")
#         print(output)
#         print("-" * 80)

# def get_prompt_template_playwright_stepdef(bdd:str) -> str:
#     template_path = os.path.join(os.path.dirname(__file__), '../../resources/playwright_stepdef_template.txt')
#     with open(template_path, 'r', encoding='utf-8') as f:
#         prompt_str = f.read()
#     return prompt_str.format(bdd=bdd)
        

if __name__ == "__main__":
    # Example usage for testing
    test_excel_path = "C:/GenAI_Related_Artifacts/April24/GHCP_MCP/resources/SampleINSBDDHistoricalContext.xlsx"  # Update with your excel path
    embedding_path = "C:/GenAI_Related_Artifacts/April24/GHCP_MCP/resources/embed_data"

    model_name = "thenlper/gte-small"
    # Test create_vector_store
    print('Vector store: ', vector_store(test_excel_path, embedding_path, model_name))

    # Test retrieve_similar_documents
    query = "User should be able to enter zip, state and find a Dentist in the Metlife application"
    n_results = 1
    #invoke the function to read user stories from excel file,
    # create a string by concatenating user story description and
    input_path = "C:/GenAI/April24/GHCP_MCP/resources/john_hancock_ac_results-all.xlsx"
    generate_prompt_for_BDD(input_path, embedding_path, 
        retrieval_directory="C:/GenAI/April24/GHCP_MCP/data/retrieval_context", n_results=n_results, model_name=model_name)
    
    # print(retrieve_context(
    #     query=query,
    #     persist_directory=embedding_path,
    #     retrieval_directory="C:/GenAI/April24/GHCP_MCP/data/retrieval_context",
    #     n_results=n_results
    # ))
