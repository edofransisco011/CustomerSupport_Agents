import json
from typing import Type, List, Dict, Any
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import os

# Define the input schema for the tool
class ConversationQueryToolInput(BaseModel):
    """Input for ConversationQueryTool."""
    query: str = Field(..., description="The search query or keywords to find relevant conversations based on tags, summary, or log content.")

class ConversationQueryTool(BaseTool):
    name: str = "Conversation Query Tool"
    description: str = (
        "Searches a knowledge base of past customer conversations. "
        "Use this tool to find examples, solutions, or information from historical support interactions "
        "based on keywords, tags, or descriptions of the customer's issue."
    )
    args_schema: Type[BaseModel] = ConversationQueryToolInput
    conversations: List[Dict[str, Any]] = []

    def __init__(self, dataset_path: str = "data/sample_conversations.json", **kwargs):
        super().__init__(**kwargs)
        # Correctly construct the absolute path to the dataset
        # Assuming this tool file is in src/customer_support_crew/tools/
        # and the data is in data/ relative to the project root
        base_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..') # up three levels to project root
        absolute_dataset_path = os.path.abspath(os.path.join(base_dir, dataset_path))

        try:
            with open(absolute_dataset_path, 'r') as f:
                self.conversations = json.load(f)
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {absolute_dataset_path}. Please ensure the path is correct.")
            self.conversations = []
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {absolute_dataset_path}.")
            self.conversations = []

    def _run(self, query: str) -> str:
        if not self.conversations:
            return "Conversation dataset is not loaded or is empty."

        query_lower = query.lower()
        relevant_conversations = []

        for conv in self.conversations:
            # Search in tags (if they exist and are a list)
            if isinstance(conv.get('tags'), list) and any(query_lower in tag.lower() for tag in conv['tags']):
                relevant_conversations.append(conv)
                continue # Move to next conversation if already found

            # Search in summary
            if query_lower in conv.get('summary', '').lower():
                relevant_conversations.append(conv)
                continue

            # Search in log (less efficient for long logs, but okay for an example)
            if query_lower in conv.get('log', '').lower():
                relevant_conversations.append(conv)
                continue
        
        if not relevant_conversations:
            return "No relevant conversations found for your query."

        # Return a summary of found conversations
        # For LLMs, it's often better to return structured data or concise summaries
        results_summary = []
        for conv in relevant_conversations[:3]: # Limit to 3 results for brevity
            results_summary.append(f"ID: {conv['id']}, Summary: {conv['summary']}")
        
        return "\n".join(results_summary) if results_summary else "No relevant conversations found."

# Example of how to instantiate if run directly (for testing)
if __name__ == '__main__':
    # This assumes you run this script from the project root or adjust the path
    # For testing, you might need to adjust the path if 'data/sample_conversations.json' isn't directly accessible
    # For example, if in src/customer_support_crew/tools:
    # tool = ConversationQueryTool(dataset_path="../../../data/sample_conversations.json")

    # More robustly, if running this file directly, determine path from this file's location:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../")) # Adjust based on actual depth
    dataset_file_path = os.path.join(project_root, "data/sample_conversations.json")

    tool = ConversationQueryTool(dataset_path=dataset_file_path)

    # Test the tool
    print("Dataset loaded:", "Yes" if tool.conversations else "No")
    print("\nSearching for 'refund':")
    print(tool._run(query="refund"))
    print("\nSearching for 'login problem':")
    print(tool._run(query="login problem"))
    print("\nSearching for 'shipping':")
    print(tool._run(query="shipping"))
    print("\nSearching for 'unknown_topic':")
    print(tool._run(query="unknown_topic"))