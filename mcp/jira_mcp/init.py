from mcp.server.fastmcp import FastMCP
from datetime import datetime
import requests
import base64
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

mcp = FastMCP("JiraMCP")

def _ensure_dir(path):
    # Check if the directory exists, if not, create it
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")
    else:
        print(f"Directory already exists: {path}")
# Jira configuration (now loaded from environment variables)
JIRA_BASE_URL = "https://ghcpmcp.atlassian.net"
JIRA_EMAIL = "rohini.thatchina@gmail.com"
JIRA_API_TOKEN = "ATCTT3xFfGN02n-GqBcO1Z-kQAUOI0juhz_kdQ7r0dstvBtoY4_Q_TDaO340tHHjaP1KUI_Ts4xpGuvTl6qxV-Xysy6AIFzTvft76j3Qn-JHoeACo-3Cz-ekL-wPhsPbc7nPNkmY6_fHU4OiBrjUjc1cTIRsDZMCU6fyCZQkk-B_F0AFcH447aM=02ACBFBD" 
PROJECT_KEY = "KEMP"

if not all([JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN, PROJECT_KEY]):
    raise ValueError("One or more Jira configuration environment variables are missing.")

# Construct Basic Auth header
auth_header = base64.b64encode(f"{JIRA_EMAIL}:{JIRA_API_TOKEN}".encode()).decode()
headers = {
    'Authorization': f'Basic {auth_header}',
    'Accept': 'application/json'
}

# Jira JQL query to fetch 'Story' issue types
jql = f"project = {PROJECT_KEY} AND issuetype = Story ORDER BY created DESC"

@mcp.tool(name="fetch_user_stories", description="Fetch user stories from Jira and save them as text files in the specified directory.")
def fetch_user_stories(directorypath: str = None):
    try:
        # Build the full URL with encoded query parameters
        url = f"{JIRA_BASE_URL}/rest/api/3/search?jql=project%20%3D%20{PROJECT_KEY}%20AND%20issuetype%20%3D%20Story%20ORDER%20BY%20created%20DESC&fields=summary%2Cstatus%2Cassignee%2Cdescription&maxResults=10"
        print(f"Requesting URL: {url}")
        print(f"Request headers: {headers}")
        response = requests.get(url, headers=headers)
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.text}")
        response.raise_for_status()  # Raise an exception for HTTP errors

        stories = response.json().get('issues', [])
        if not stories:
            print("No user stories found.")
            return

        # Determine the directory to store user stories
        user_stories_dir = directorypath or "UserStories"
        _ensure_dir(user_stories_dir)

        for issue in stories:
            fields = issue.get('fields', {})
            user_story_id = issue.get("key", "NO-KEY")
            summary = fields.get('summary', 'No summary')
            status = fields.get('status', {}).get('name', 'No status')
            assignee = fields.get('assignee')
            assignee_name = assignee.get('displayName', 'Unassigned') if assignee else 'Unassigned'
            description = fields.get('description')
            desc_text = extract_description_text(description) if description else 'No description'

            # Remove non-ASCII characters from desc_text before writing (for Windows compatibility)
            #safe_desc_text = desc_text.encode('ascii', 'ignore').decode('ascii')

            print(f"    {user_story_id}: {summary}")
            print(f"   Status: {status}")
            print(f"   Assignee: {assignee_name}\n")
            print(f"   Description: {desc_text}\n")

            # Save user story to a file
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            output_filename = f"{user_story_id}_{timestamp}.txt"
            output_path = os.path.join(user_stories_dir, output_filename)
            # Use UTF-8 encoding and ignore errors to handle any special characters
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"User Story ID: {user_story_id}\n")
                f.write(f"Summary: {summary}\n")
                f.write(f"Description: {desc_text}\n")
            print(f"   User story saved to {output_path}\n")

        print("User stories fetched and saved successfully.")

    except requests.exceptions.RequestException as e:
        print(f"HTTP request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Error response content: {e.response.text}")
    except Exception as e:
        print(f"Failed to fetch user stories: {e}")

def extract_description_text(description):
    if not description or 'content' not in description:
        return 'No description'
    def extract_content(content_list):
        texts = []
        for item in content_list:
            if 'text' in item:
                texts.append(item['text'])
            if 'content' in item:
                texts.extend(extract_content(item['content']))
        return texts
    return '\n'.join(extract_content(description['content']))



if __name__ == "__main__":
    fetch_user_stories("C:/GenAI_Related_Artifacts/April24/GHCP_MCP/data/UserStories")