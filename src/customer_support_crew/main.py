import os
import sys
import argparse
import logging
import configparser
from dotenv import load_dotenv
import datetime # For timestamp
import re # For sanitizing filenames

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file at the very beginning
# Check multiple possible locations for the .env file
def load_env_file():
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '..', '..', '.env'),  # From src/customer_support_crew
        os.path.join(os.getcwd(), '.env'),  # From current working directory
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"Loading environment variables from {path}")
            load_dotenv(dotenv_path=path)
            return True
    
    logger.warning("No .env file found. Please ensure environment variables are set manually.")
    return False

load_env_file()

from customer_support_crew.crew import CustomerSupportCrew

def sanitize_filename(name_base: str, max_length: int = 60) -> str:
    """Sanitizes a string to be a valid filename component."""
    # Remove non-alphanumeric characters (except spaces, hyphens, underscores)
    name = re.sub(r'[^\w\s-]', '', name_base)
    # Replace whitespace and multiple hyphens/underscores with a single underscore
    name = re.sub(r'[-\s]+', '_', name).strip('_')
    # Truncate to max_length
    return name[:max_length]

def get_config(config_path=None):
    """Load configuration from a config file if it exists, otherwise use defaults"""
    config = configparser.ConfigParser()
    
    # Set default values
    config['DEFAULT'] = {
        'OutputDirectory': 'output',
        'ModelProvider': 'nvidia_nim',
        'ModelName': 'deepseek-ai/deepseek-r1',
        'DatasetPath': 'data/sample_conversations.json',
    }
    
    # Try to load from config file if it exists
    if config_path and os.path.exists(config_path):
        try:
            config.read(config_path)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
    
    return config

def validate_required_env_vars():
    """Validate that all required environment variables are set"""
    required_vars = ["NVIDIA_NIM_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set them in your .env file or environment.")
        return False
    return True

def run(customer_query=None, config_path=None):
    """
    Run the customer support crew.
    
    Args:
        customer_query (str, optional): The customer query to process
        config_path (str, optional): Path to the configuration file
    """
    # Load configuration
    config = get_config(config_path)
    
    # Define project root and output directory
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_dir, "..", "..")) # Up two levels
    output_dir = os.path.join(project_root, config['DEFAULT']['OutputDirectory'])
    
    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Project root: {project_root}")
        logger.info(f"Output directory: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directory: {e}")
        return

    # Get customer query from command line if not provided
    if not customer_query:
        # Default queries for example/testing
        example_queries = [
            "I need a refund for my order.",
            "I can't log in to my account.",
            "How long does shipping take for Product X?",
            "I need to cancel my subscription",
            "My payment failed for my subscription, what should I do? It's urgent!"
        ]
        
        # If running in interactive mode and no query provided
        if sys.stdin.isatty():
            print("\nExample queries:")
            for i, query in enumerate(example_queries, 1):
                print(f"{i}. {query}")
            print("\nEnter your query or select a number from the examples above:")
            user_input = input("> ").strip()
            
            # Check if user selected an example query number
            if user_input.isdigit() and 1 <= int(user_input) <= len(example_queries):
                customer_query = example_queries[int(user_input)-1]
            else:
                customer_query = user_input
        else:
            # Default query for non-interactive mode
            customer_query = "My payment failed for my subscription, what should I do? It's urgent!"
    
    logger.info(f"Processing customer query: \"{customer_query}\"")

    # Generate a unique filename stem for the output
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_query_part = sanitize_filename(customer_query)
    # This is the name that will be used in the {generated_filename} placeholder
    filename_stem = f"support_response_{timestamp}_{sanitized_query_part}"

    inputs = {
        'customer_query': customer_query,
        'generated_filename': filename_stem # Pass the filename stem
    }
    
    logger.info(f"Generated filename stem for output: \"{filename_stem}\"")
    
    try:
        # Create the crew with configuration
        support_crew_instance = CustomerSupportCrew(
            dataset_path=config['DEFAULT']['DatasetPath']
        )
        result = support_crew_instance.crew().kickoff(inputs=inputs)
        
        logger.info("Customer Support Crew completed processing query.")
        
        # Check if the output file was created
        expected_file_path = os.path.join(output_dir, f"{filename_stem}.md")
        if os.path.exists(expected_file_path):
            logger.info(f"SUCCESS: Output file created at {expected_file_path}")
            print(f"\nResponse saved to: {expected_file_path}")
            
            # Optionally show a preview of the response
            print("\n--- Response Preview ---")
            with open(expected_file_path, 'r', encoding='utf-8') as f:
                preview = f.read(500)  # Show first 500 chars
                print(preview + ("..." if len(preview) == 500 else ""))
            print("--- End of Preview ---")
        else:
            logger.warning(f"Output file not found at {expected_file_path}")
            print("\n--- Agent's Response ---")
            print(result.raw)
            print("--- End of Response ---")
            
    except Exception as e:
        logger.error(f"Error running customer support crew: {e}", exc_info=True)
        print(f"Error: {e}")

def main():
    """Command line interface for the customer support application"""
    parser = argparse.ArgumentParser(description='Customer Support AI Assistant')
    parser.add_argument('--query', '-q', type=str, help='Customer query to process')
    parser.add_argument('--config', '-c', type=str, help='Path to configuration file')
    args = parser.parse_args()
    
    # Validate environment variables
    if not validate_required_env_vars():
        sys.exit(1)
    
    # Run the application
    run(customer_query=args.query, config_path=args.config)

if __name__ == "__main__":
    main()