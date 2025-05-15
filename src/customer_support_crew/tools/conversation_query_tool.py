import json
from typing import Type, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import os
import logging
import re
import functools
from collections import Counter

# Configure logging
logger = logging.getLogger(__name__)

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
    
    # Cache to store previous query results
    _query_cache: Dict[str, List[Dict[str, Any]]] = {}
    
    # Maximum cache size
    _MAX_CACHE_SIZE: int = 100

    def __init__(self, dataset_path: str = "data/sample_conversations.json", cache_size: int = 100, **kwargs):
        """
        Initialize the ConversationQueryTool.
        
        Args:
            dataset_path (str): Path to the dataset JSON file
            cache_size (int): Maximum number of queries to cache
            **kwargs: Additional arguments to pass to the parent class
        """
        super().__init__(**kwargs)
        self._MAX_CACHE_SIZE = cache_size
        self._query_cache = {}
        
        # Try to load the dataset from the provided path
        self._load_dataset(dataset_path)
    
    def _load_dataset(self, dataset_path: str) -> None:
        """
        Load the dataset from the specified path.
        
        Args:
            dataset_path (str): Path to the dataset JSON file
        """
        # Construct the absolute path to the dataset if needed
        try:
            # If the path is not absolute, try to resolve it
            if not os.path.isabs(dataset_path):
                # Check if file exists as is
                if not os.path.exists(dataset_path):
                    # Try to resolve relative to the module directory
                    current_script_dir = os.path.dirname(os.path.abspath(__file__))
                    project_root = os.path.abspath(os.path.join(current_script_dir, "..", "..", ".."))
                    dataset_path = os.path.join(project_root, dataset_path)
                    
                    # If still not found, try one level up
                    if not os.path.exists(dataset_path):
                        alt_project_root = os.path.abspath(os.path.join(current_script_dir, "..", ".."))
                        dataset_path = os.path.join(alt_project_root, dataset_path)
            
            logger.info(f"Attempting to load dataset from: {dataset_path}")
            
            with open(dataset_path, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
                logger.info(f"Successfully loaded {len(self.knowledge_base)} entries from dataset")
                
        except FileNotFoundError:
            logger.error(f"Dataset file not found at {dataset_path}")
            self.knowledge_base = []
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in dataset file at {dataset_path}")
            self.knowledge_base = []
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            self.knowledge_base = []

    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess the query for better matching.
        
        Args:
            query (str): The search query
            
        Returns:
            str: The preprocessed query
        """
        # Convert to lowercase
        query = query.lower()
        
        # Remove punctuation
        query = re.sub(r'[^\w\s]', ' ', query)
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text (str): The text to tokenize
            
        Returns:
            List[str]: List of tokens (words)
        """
        # Simple word tokenization
        return [word for word in re.split(r'\W+', text.lower()) if word]

    def _calculate_relevance_score(self, entry: Dict[str, Any], query_tokens: List[str]) -> float:
        """
        Calculate a relevance score for the entry based on the query.
        
        Args:
            entry (Dict[str, Any]): The knowledge base entry
            query_tokens (List[str]): The tokenized query
            
        Returns:
            float: The relevance score (higher is more relevant)
        """
        score = 0.0
        entry_type = entry.get('type', '')
        
        # Check for exact matches in important fields (higher weight)
        important_fields = ['id', 'tags', 'summary', 'title']
        for field in important_fields:
            if field in entry:
                field_value = entry[field]
                
                # Handle list fields like tags
                if isinstance(field_value, list):
                    # For each tag, check if it contains any of the query tokens
                    for item in field_value:
                        if isinstance(item, str):
                            item_tokens = self._tokenize(item)
                            for query_token in query_tokens:
                                if query_token in item_tokens:
                                    score += 2.0  # Higher weight for tag matches
                
                # Handle string fields
                elif isinstance(field_value, str):
                    field_tokens = self._tokenize(field_value)
                    # Count matching tokens
                    matches = sum(token in field_tokens for token in query_tokens)
                    # Weight by field importance
                    if field == 'id':
                        score += matches * 1.0
                    elif field == 'summary':
                        score += matches * 2.0
                    elif field == 'title':
                        score += matches * 3.0  # Title matches are highly relevant
        
        # Type-specific scoring
        if entry_type == "conversation_example":
            if 'log' in entry and isinstance(entry['log'], str):
                log_tokens = self._tokenize(entry['log'])
                # Check for query token presence in log
                matches = sum(token in log_tokens for token in query_tokens)
                score += matches * 1.0
                
        elif entry_type == "guideline":
            # Check description and examples
            if 'description' in entry and isinstance(entry['description'], str):
                desc_tokens = self._tokenize(entry['description'])
                matches = sum(token in desc_tokens for token in query_tokens)
                score += matches * 1.5
                
            if 'examples' in entry and isinstance(entry['examples'], list):
                for example in entry['examples']:
                    if isinstance(example, str):
                        example_tokens = self._tokenize(example)
                        matches = sum(token in example_tokens for token in query_tokens)
                        score += matches * 1.0
        
        # Boost score if the query specifically mentions the entry type
        type_boost = 0
        if "guideline" in query_tokens and entry_type == "guideline":
            type_boost = 5.0
        elif any(word in query_tokens for word in ["conversation", "example"]) and entry_type == "conversation_example":
            type_boost = 5.0
            
        return score + type_boost

    @functools.lru_cache(maxsize=100)
    def _search_knowledge_base(self, query: str) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for entries matching the query.
        
        Args:
            query (str): The preprocessed search query
            
        Returns:
            List[Dict[str, Any]]: List of matching entries sorted by relevance
        """
        # Check if query is in cache
        if query in self._query_cache:
            logger.info(f"Query cache hit for: {query}")
            return self._query_cache[query]
            
        if not self.knowledge_base:
            logger.warning("Knowledge base is empty")
            return []
            
        # Process the query
        query_tokens = self._tokenize(query)
        
        # Calculate scores for entries
        scored_entries = []
        for entry in self.knowledge_base:
            score = self._calculate_relevance_score(entry, query_tokens)
            if score > 0:  # Only include entries with a positive score
                scored_entries.append((entry, score))
        
        # Sort by score (highest first)
        scored_entries.sort(key=lambda x: x[1], reverse=True)
        
        # Extract just the entries
        relevant_entries = [entry for entry, _ in scored_entries]
        
        # Store in cache if not empty
        if relevant_entries:
            # If cache is full, remove oldest entry
            if len(self._query_cache) >= self._MAX_CACHE_SIZE:
                oldest_query = next(iter(self._query_cache))
                del self._query_cache[oldest_query]
                
            self._query_cache[query] = relevant_entries
            
        return relevant_entries

    def _format_entry_for_output(self, entry: Dict[str, Any], index: int, total: int) -> str:
        """
        Format a knowledge base entry for output.
        
        Args:
            entry (Dict[str, Any]): The knowledge base entry
            index (int): The index of the entry in the results
            total (int): The total number of results
            
        Returns:
            str: The formatted entry
        """
        entry_type = entry.get('type', 'unknown')
        entry_id = entry.get('id', 'N/A')
        
        if entry_type == "guideline":
            title = entry.get('title', 'No Title')
            summary = entry.get('summary', 'No Summary')
            description = entry.get('description', '')
            
            result = f"Found Guideline (ID: {entry_id}) [{index+1}/{total}]\n"
            result += f"Title: {title}\n"
            result += f"Summary: {summary}\n"
            
            if description:
                # Show truncated description if it's long
                if len(description) > 150:
                    result += f"Description: {description[:150]}...\n"
                else:
                    result += f"Description: {description}\n"
            
            # Show examples if available
            if 'examples' in entry and entry['examples']:
                result += "Examples:\n"
                # Show up to 3 examples
                for i, example in enumerate(entry['examples'][:3]):
                    result += f"  - {example}\n"
                
                # Indicate if there are more examples
                if len(entry['examples']) > 3:
                    result += f"  ... and {len(entry['examples']) - 3} more examples\n"
                    
        else:  # conversation_example
            summary = entry.get('summary', 'No Summary')
            tags = entry.get('tags', [])
            
            result = f"Found Conversation Example (ID: {entry_id}) [{index+1}/{total}]\n"
            result += f"Summary: {summary}\n"
            
            if tags:
                result += f"Tags: {', '.join(tags)}\n"
            
            # Show snippet of the log
            if 'log' in entry and entry['log']:
                log_lines = entry['log'].split('\n')
                result += "Conversation Snippet:\n"
                # Show up to 4 lines of the log
                for line in log_lines[:4]:
                    result += f"  {line}\n"
                    
                # Indicate if the log is longer
                if len(log_lines) > 4:
                    result += f"  ... and {len(log_lines) - 4} more lines\n"
        
        return result

    def _run(self, query: str) -> str:
        """
        Execute the tool with the given query.
        
        Args:
            query (str): The search query
            
        Returns:
            str: The search results formatted as a string
        """
        try:
            if not self.knowledge_base:
                return "Knowledge base is not loaded or is empty."
                
            # Preprocess the query
            processed_query = self._preprocess_query(query)
            logger.info(f"Searching knowledge base for: {processed_query}")
            
            # Search the knowledge base
            relevant_entries = self._search_knowledge_base(processed_query)
            
            if not relevant_entries:
                return "No relevant entries found for your query in the knowledge base."
                
            # Format the results
            results_output = []
            max_results = min(5, len(relevant_entries))  # Display up to 5 results
            
            for i in range(max_results):
                entry = relevant_entries[i]
                formatted_entry = self._format_entry_for_output(entry, i, max_results)
                results_output.append(formatted_entry)
                
            # Add a note if there are more results
            if len(relevant_entries) > max_results:
                results_output.append(f"\n...and {len(relevant_entries) - max_results} more entries found. You can refine your query to see different results.")
                
            return "\n\n---\n\n".join(results_output)
            
        except Exception as e:
            logger.error(f"Error in ConversationQueryTool: {str(e)}")
            return f"An error occurred while searching: {str(e)}"

# Example of how to instantiate if run directly (for testing)
if __name__ == '__main__':
    # Set up logging for testing
    logging.basicConfig(level=logging.INFO)
    
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