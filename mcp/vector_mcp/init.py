import os
from datetime import datetime
from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import csv_loader
import pandas as pd

MODEL_NAME = "thenlper/gte-small"
mcp = FastMCP(name="UserStoryToBDD", description="Generate BDD prompts from user stories using context retrieval.")

def _ensure_dir(path):
    if os.path.isfile(path):
        raise ValueError(f"Path '{path}' is a file, not a directory. Please provide a directory path.")
    os.makedirs(path, exist_ok=True)

def _excel_to_csv(excel_path):
    csv_path = os.path.splitext(excel_path)[0] + ".csv"
    df = pd.read_excel(excel_path)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path

@mcp.tool(
    name="vector_store",
    description="Create a vector store from a context Excel file as the first step in generating BDD prompts for user stories. This prepares context embeddings for later retrieval during BDD prompt generation. If embeddings already exist, you can choose to overwrite them by setting overwrite=True."
)
def vector_store(contextpath: str, overwrite: bool = False):
    """
    To generate BDD prompt from User story, first step is to create a vector store from an Excel file and store embeddings.
    If embeddings already exist, ask the user if they want to overwrite or use existing embeddings.
    inputs:
    - contextpath: Path to the Excel file containing context data.
    - overwrite: Whether to overwrite existing embeddings (default False).
    outputs:
    - Returns a string indicating the number of documents created and persisted in the vector store, or a message if embeddings already exist.
    Steps:
    1. If the context file is an Excel file, convert it to CSV.
    2. Check if embeddings already exist. If so, and overwrite is False, return a message to the user.
    3. If overwrite is True or embeddings do not exist, create embeddings and persist them.
    """
    
    # Ensure absolute path for contextpath
    contextpath = os.path.abspath(contextpath)
    # Set embedding_path to a fixed MCP server location
    embedding_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../resources/embed_data'))
    _ensure_dir(embedding_path)
    # Check for existing embeddings (e.g., chroma.sqlite3 file)
    chroma_db_file = os.path.join(embedding_path, 'chroma.sqlite3')
    if os.path.exists(chroma_db_file) and not overwrite:
        return ("Vector embeddings already exist. If you want to overwrite them, call this tool again with overwrite=True.")
    if contextpath.lower().endswith((".xls", ".xlsx")):
        contextpath = _excel_to_csv(contextpath)
    loader = csv_loader.CSVLoader(file_path=contextpath, encoding="utf-8")
    documents = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    vector_store = Chroma.from_documents(documents, embeddings, persist_directory=embedding_path)
    vector_store.persist()
    return f"Vector store created and persisted with {len(documents)} documents."

#@mcp.tool(name="retrieve_context", description="Retrieve similar documents for a query and save to file.")
def retrieve_context(query: str, persist_directory: str, retrieval_directory: str, n_results: int = 1):
    """
    To generate BDD prompt from Userstory, after creating vector embeddings, retrieve similar documents from a vector store based on a query and save the context to a file.
    inputs:
    - query: The query string to search for similar documents.
    - persist_directory: Directory where the vector store is persisted.
    - retrieval_directory: Directory where the retrieved context will be saved.
    - model_name: Name of the HuggingFace model to use for embeddings.
    - n_results: Number of similar documents to retrieve (default is 1).
    outputs:
    - Returns a string containing the concatenated page content of the retrieved documents.
    Steps:
    1. Ensure the retrieval directory exists.
    2. Load the embeddings model.
    3. Load the vector store from the specified persist directory.
    4. Perform similarity search using the query.
    5. Concatenate the page content of the retrieved documents.
    6. Save the context string to a file in the retrieval directory with a timestamp.
    7. Return the context string.
    """
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retrieved_docs = vector_store.similarity_search(query, k=n_results)
    context_str = "\n\n".join(doc.page_content for doc in retrieved_docs)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_path = os.path.join(retrieval_directory, f"TestCase_Context_{timestamp}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(context_str)
    return context_str


@mcp.tool(
    name="generate_prompt_for_BDD",
    description="Generate BDD prompts for each user story in an Excel file by retrieving relevant context from the vector store. This is the second step after creating the vector store."
)
def generate_prompt_for_BDD(input_excel_path: str):
    """
    To generate BDD, after creating vector embeddings, retrieve similar context from db, generate BDD prompts for each user story in an Excel file.
    inputs:
    - input_excel_path: Path to the input Excel file containing user stories.
    - persist_directory: Directory where the vector store is persisted.
    - retrieval_directory: Directory where the retrieved context and prompts will be saved.
    - model_name: Name of the HuggingFace model to use for embeddings.
    - n_results: Number of similar documents to retrieve for each user story (default is 1).
    outputs:
    - Saves BDD prompts to the specified retrieval directory.
    Steps:
    1. Read and preprocess the Excel file: drop rows with missing user stories or acceptance criteria, strip whitespace.
    2. For each user story, retrieve context and generate BDD prompt.
    3. Save BDD prompts to the specified retrieval directory.
    """
    # use path of input excel path to create retrieval directory
    input_excel_path = os.path.abspath(input_excel_path)
    retrieval_directory = os.path.join(os.path.dirname(input_excel_path), "retrieval_context")
    #retrieval_directory = os.path.join(os.path.dirname(__file__), "../../resources/retrieval_context")
    persist_directory = os.path.join(os.path.dirname(__file__), "../../resources/embed_data")
    model_name = MODEL_NAME
    n_results = 1  # Default number of results to retrieve
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
    
        print(f"Generated BDD Prompt for User Story {idx + 1}:")
        print(output)
        print("-" * 80)


def get_prompt_template(user_story: str, context: str) -> str:
    """
    Generate a prompt template for BDD based on the user story and context.
    inputs:
    - user_story: The user story string to be included in the prompt.
    - context: The context string to be included in the prompt.
    outputs:
    - Returns a formatted prompt string.
    Steps:
    1. Read the prompt template from a file.
    2. Format the template with the user story and context.
    3. If context is empty, pass only the user story.
    4. Return the formatted prompt string.
    """
    # if context is empty, invoke prompt_template_no_context.txt, else invoke prompt_template.txt
    template_path = os.path.join(os.path.dirname(__file__), '../../resources/prompt_template.txt')
    if not context.strip():
        template_path = os.path.join(os.path.dirname(__file__), '../../resources/prompt_template_no_context.txt')
    print('template_path:', template_path)
    with open(template_path, 'r', encoding='utf-8') as f:
        prompt_str = f.read()
    # Format the prompt string with user story and context
    if context.strip():
        # If context is provided, include it in the prompt
        return prompt_str.format(user_story=user_story, context=context)

    return prompt_str.format(user_story=user_story, context='')
    
    

        
#@mcp.tool(name="generate_prompt_for_BDD_from_txt", description="Generate BDD Prompt for each user story in a directory of text files.")
def generate_prompt_for_BDD_from_txt(userstories_dir: str, persist_directory: str, retrieval_directory: str, model_name: str, n_results: int = 1):
    """
    Generate BDD prompts for each user story in a directory of text files.
    inputs:
    - userstories_dir: Directory containing text files with user stories.
    - persist_directory: Directory where the vector store is persisted.
    - retrieval_directory: Directory where the retrieved context and prompts will be saved.
    - model_name: Name of the HuggingFace model to use for embeddings.
    - n_results: Number of similar documents to retrieve for each user story (default is 1).
    outputs:
    - Saves BDD prompts to the specified retrieval directory.
    Steps:
    1. Ensure the retrieval directory exists.
    2. Ensure the prompts directory exists inside the retrieval directory.
    3. List all text files in the user stories directory.
    4. For each text file, read the user story, retrieve context, and generate BDD prompt.
    5. Save the generated prompt to a file in the prompts directory.
    """
    _ensure_dir(retrieval_directory)
    prompts_dir = os.path.join(retrieval_directory, "Prompts")
    _ensure_dir(prompts_dir)
    txt_files = [f for f in os.listdir(userstories_dir) if f.lower().endswith('.txt')]
    for idx, filename in enumerate(txt_files):
        file_path = os.path.join(userstories_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            user_story = f.read().strip()
        query = user_story
        context_str = retrieve_context(
            query=query,
            persist_directory=persist_directory,
            retrieval_directory=retrieval_directory,
            model_name=model_name,
            n_results=n_results
        )
        output = get_prompt_template(query, context_str)
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename = filename.split("_")[0]
        output_filename = filename + f"_{timestamp}.txt"
        output_path = os.path.join(prompts_dir, output_filename)
        with open(output_path, "w", encoding="utf-8") as out_f:
            out_f.write(output)
        print(f"Generated BDD for User Story file {filename}:")
        print(output)
        print("-" * 80)

@mcp.tool(
    name="generate_bdd_from_userstories_no_context",
    description="To generate BDD from only user stories, first step is to generate BDD prompts directly from user stories in an Excel file without using any context. Use this tool when no context is provided for retrieval."
)
def generate_bdd_from_userstories_no_context(userstories_excel_path: str):
    """
    Generate BDD prompts directly from user stories in an Excel file without using any context.
    Use this function when no context is available or required for retrieval.
    Prompts for BDD can be used to generate new BDD feature files.
    Intended for use by copilot agent to write BDD using the prompts and save it to a new feature file.
    inputs:
    - userstories_excel_path: Path to the user stories Excel file.
    outputs:
    - Writes BDD prompts which can be used to generate BDD feature files.
    Steps:
    1. Read user stories from the Excel file.
    2. For each user story, generate a BDD prompt using only the user story description and acceptance criteria.
    3. Writes BDD prompts for each use story which can be used to generate BDD feature files.
    4. Copilot agent can use these prompts to write BDD and save it to a new feature file.
    """

    # Step 1: Read user stories Excel
    df = pd.read_excel(userstories_excel_path)
    df = df.dropna(subset=["Description", "AcceptanceCriteria"])
    df["Description"] = df["Description"].astype(str).str.strip()
    df["AcceptanceCriteria"] = df["AcceptanceCriteria"].astype(str).str.strip()

    results = []
    for idx, row in df.iterrows():
        user_story_id = row.get("UserStoryID", idx + 1)
        print(f"Processing User Story {user_story_id}...")
        query = f"{row['Description']} {row.get('FunctionalRequirements', '')} {row['AcceptanceCriteria']} {row.get('Assumptions', '')}"
        
        # this can be used to generate BDD
        output = get_prompt_template(query, "")

        results.append({"user_story_id": user_story_id, "prompt": output})

    return results
    

@mcp.tool(
    name="generate_bdd_for_user_stories",
    description="To Generate BDD for user stories, first create a vector store from a context Excel file and then generate BDD prompts for each user story in an Excel file. This tool automates the full BDD generation pipeline from context and user stories. If embeddings already exist, you will be prompted to confirm overwriting."
)
def generate_bdd_for_user_stories(context_excel_path: str, userstories_excel_path: str, overwrite: bool = False):
    """
    Orchestrates the BDD generation process:
    1. Creates a vector store from the provided context Excel file (with overwrite confirmation if embeddings exist).
    2. Generates BDD prompts for each user story in the user stories Excel file using the created vector store.
    inputs:
    - context_excel_path: Path to the context Excel file.
    - userstories_excel_path: Path to the user stories Excel file.
    - overwrite: Whether to overwrite existing embeddings (default False).
    outputs:
    - Runs the full BDD prompt generation pipeline and saves results to output directories, or prompts user to confirm overwriting embeddings.
    """
    result = vector_store(context_excel_path, overwrite=overwrite)
    # If embeddings exist and not overwriting, notify but continue to prompt generation
    if isinstance(result, str) and "already exist" in result and not overwrite:
        print(result)  # Log message for user
        # Continue to prompt generation
    generate_prompt_for_BDD(userstories_excel_path)
    return "BDD prompts generated for user stories."



def get_prompt_template_step_def(bdd: str) -> str:
    """
    Generate a prompt template for BDD step definition for the BDD.
    inputs:
    - bdd: The bdd string to be included in the prompt.
    outputs:
    - Returns a formatted prompt string.
    Steps:
    1. Read the prompt template from a file.
    2. Format the template with the bdd.
    3. Return the formatted prompt string.
    """
    
    # Read the prompt template from a file and format it with user story and context
    template_path = os.path.join(os.path.dirname(__file__), '../../resources/prompt_template_step_def.txt')
    print('template_path:', template_path)
    with open(template_path, 'r', encoding='utf-8') as f:
        prompt_str = f.read()
    
    # Return the formatted prompt string
    return prompt_str.format(bdd=bdd)

@mcp.tool(
    name="generate_typescript_step_definitions_from_feature_file",
    description="Generate step definitions in typescript for a feature file or BDD string. Accepts either a file path to a .feature file or the BDD content directly. This tool creates step definitions based on the BDD and saves them to a file. Returns the output file path. Also responds to requests like 'generate step definition for a feature file', 'create step definition', or 'step definition for feature file'."
)
def generate_step_definitions(bdd_or_path: str):
    """
    Generate step definitions for the given feature file or BDD string and save to a file.
    inputs:
    - bdd_or_path: Path of the feature file or the BDD string itself.
    outputs:
    - Copilot agent saves the generated step definitions to a file.
    """
    bdd_content = None
    # Check if input is a file path
    if os.path.isfile(bdd_or_path):
        with open(bdd_or_path, 'r', encoding='utf-8') as f:
            bdd_content = f.read().strip()
    else:
        # Assume input is BDD string
        bdd_content = bdd_or_path.strip()
        # Use a default retrieval directory
    output = get_prompt_template_step_def(bdd_content)
    return output

if __name__ == "__main__":
    # test the vector_store function
    context_path = "C:/GenAI_Related_Artifacts/April24/GHCP_MCP/resources/KemperContext.xlsx"  
    
    embedding_path = "C:/GenAI_Related_Artifacts/April24/GHCP_MCP/resources/Embed_data"  
    #vector_store(context_path)
    # # test the generate_prompt_for_BDD function
    input_userstories_path = "C:/GenAI_Related_Artifacts/April24/GHCP_MCP/resources/KemperUserstoryInput.xlsx"  
    retrieval_directory = "C:/GenAI_Related_Artifacts/April24/GHCP_MCP/resources/Retrieval_context"  
    #generate_prompt_for_BDD(input_userstories_path)
    #generate_bdd_for_user_stories(context_path, input_userstories_path, overwrite=False)
    bdd_path = "C:/GenAI_Related_Artifacts/April24/GHCP_MCP/resources/FindAnAgent.feature"
    # test the generate_step_definitions function
    #generate_step_definitions(bdd_path)   
    generate_bdd_from_userstories_no_context(input_userstories_path)
