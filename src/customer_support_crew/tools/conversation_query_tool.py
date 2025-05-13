import json
from typing import Type, List, Dict, Any
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import os

# Define the input schema for the tool
class ConversationQueryToolInput(BaseModel):
    """Input for ConversationQueryTool."""
    query: str = Field(..., description="The search query or keywords to find relevant conversations or guidelines. Specify if you're looking for 'guidelines' or 'conversations' if relevant.")

class ConversationQueryTool(BaseTool):
    name: str = "Knowledge Base Query Tool"
    description: str = (
        "Searches a knowledge base of past customer conversations and support guidelines. "
        "Use this tool to find examples, solutions, best practices, or information from historical support interactions "
        "or established guidelines based on keywords, tags, or descriptions of the customer's issue or your query."
    )
    args_schema: Type[BaseModel] = ConversationQueryToolInput
    knowledge_base: List[Dict[str, Any]] = []

    def __init__(self, dataset_path: str = "data/sample_conversations.json", **kwargs):
        super().__init__(**kwargs)
        # Construct the absolute path to the dataset
        # Assumes this tool file is in src/customer_support_crew/tools/
        # and data/ is relative to the project root.
        # To make it robust:
        # current_script_dir = os.path.dirname(os.path.abspath(__file__))
        # project_root = os.path.abspath(os.path.join(current_script_dir, "..", "..", "..")) # Adjust as per your structure
        # absolute_dataset_path = os.path.join(project_root, dataset_path)
        
        # Simpler path construction assuming a standard project layout where 'data' is at the root
        # and this script might be run from various locations during execution by CrewAI.
        # Using a path relative to a known root or an absolute path is best.
        # For this example, we assume the 'data' folder is at the same level as 'src' or the CWD is project root.
        # The path in __init__ of crew.py already handles this well, let's keep consistency if possible.
        # For direct tool testing, you might need to adjust.
        # Fallback for direct testing if not in project root:
        if not os.path.exists(dataset_path):
             base_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..')
             dataset_path = os.path.abspath(os.path.join(base_dir, dataset_path))


        try:
            with open(dataset_path, 'r', encoding='utf-8') as f: # Added encoding
                self.knowledge_base = json.load(f)
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {dataset_path}. Please ensure the path is correct.")
            self.knowledge_base = []
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {dataset_path}.")
            self.knowledge_base = []

    def _search_entry(self, entry: Dict[str, Any], query_lower: str) -> bool:
        """Checks if a single entry matches the query."""
        # Search in 'id'
        if query_lower in entry.get('id', '').lower():
            return True
        # Search in 'tags'
        if isinstance(entry.get('tags'), list) and any(query_lower in tag.lower() for tag in entry['tags']):
            return True
        # Search in 'summary'
        if query_lower in entry.get('summary', '').lower():
            return True
        # Search in 'language' (e.g., query "english guidelines")
        if query_lower in entry.get('language', '').lower():
             # If query is just a language, might be too broad, but included for completeness
            pass # Let other fields determine relevance unless query is specific like "english guidelines"

        entry_type = entry.get('type')
        if entry_type == "conversation_example":
            if query_lower in entry.get('log', '').lower():
                return True
        elif entry_type == "guideline":
            if query_lower in entry.get('title', '').lower():
                return True
            if query_lower in entry.get('description', '').lower():
                return True
            if isinstance(entry.get('examples'), list) and \
               any(query_lower in ex.lower() for ex in entry['examples']):
                return True
        return False

    def _run(self, query: str) -> str:
        if not self.knowledge_base:
            return "Knowledge base is not loaded or is empty."

        query_lower = query.lower()
        relevant_entries = []

        # Simple filtering based on query terms
        search_type = None
        if "guideline" in query_lower:
            search_type = "guideline"
        elif "conversation" in query_lower or "example" in query_lower:
            search_type = "conversation_example"

        for entry in self.knowledge_base:
            # If a specific type is mentioned in the query, filter by it first
            if search_type and entry.get('type') != search_type:
                continue
            
            if self._search_entry(entry, query_lower):
                relevant_entries.append(entry)
        
        if not relevant_entries:
            return "No relevant entries found for your query in the knowledge base."

        # Return a formatted summary of found entries
        # Limit to a few results to avoid overwhelming the LLM context
        results_output = []
        for i, entry in enumerate(relevant_entries[:3]): # Display top 3 matches
            entry_type = entry.get('type', 'N/A')
            entry_id = entry.get('id', 'N/A')
            summary = entry.get('summary', 'No summary available.')
            title = entry.get('title', '') # Relevant for guidelines

            if entry_type == "guideline":
                result_str = f"Found Guideline (ID: {entry_id}):\nTitle: {title}\nSummary: {summary}\n"
                if entry.get('examples'):
                    result_str += "Examples:\n" + "\n".join([f"  - {ex}" for ex in entry['examples'][:2]]) # Show 1-2 examples
            else: # conversation_example
                result_str = f"Found Conversation Example (ID: {entry_id}):\nSummary: {summary}\n"
                # Optionally, include a snippet of the log for conversation examples
                log_snippet = entry.get('log', '').split('\n')[:3] # First 3 lines of log
                if log_snippet:
                    result_str += "Log Snippet:\n" + "\n".join([f"  {line}" for line in log_snippet])

            results_output.append(result_str)
            if i == 2 and len(relevant_entries) > 3 : # if more than 3 results, indicate it
                results_output.append(f"\n...and {len(relevant_entries) - 3} more entries found.")


        return "\n\n---\n\n".join(results_output) if results_output else "No relevant entries found for your query."

# Example of how to instantiate if run directly (for testing)
if __name__ == '__main__':
    # Determine path from this file's location for robust testing
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming structure: project_root/src/customer_support_crew/tools/this_file.py
    # and project_root/data/dataset.json
    project_root = os.path.abspath(os.path.join(current_dir, "../../../")) 
    dataset_file_path = os.path.join(project_root, "data/sample_conversations.json")

    # Check if the path construction is correct by printing it
    print(f"Attempting to load dataset from: {dataset_file_path}")

    tool = ConversationQueryTool(dataset_path=dataset_file_path)

    print("\nKnowledge Base loaded:", "Yes" if tool.knowledge_base else "No", f"({len(tool.knowledge_base)} entries)")
    
    print("\n--- Searching for 'refund' (conversation):")
    print(tool._run(query="refund conversation"))
    
    print("\n--- Searching for 'polite language' (guideline):")
    print(tool._run(query="polite language guideline"))

    print("\n--- Searching for 'login':")
    print(tool._run(query="login"))

    print("\n--- Searching for 'follow up':")
    print(tool._run(query="follow up guideline"))

    print("\n--- Searching for 'unknown_topic':")
    print(tool._run(query="unknown_topic"))

    print("\n--- Searching for 'empathy':")
    print(tool._run(query="empathy"))