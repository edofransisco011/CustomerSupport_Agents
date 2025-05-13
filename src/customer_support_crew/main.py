import os
from dotenv import load_dotenv
import datetime # For timestamp
import re # For sanitizing filenames

# Load environment variables from .env file at the very beginning
# Assuming .env is in the project root (customer_support_crew/)
# If main.py is in src/customer_support_crew, .env is ../../.env
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=dotenv_path)


from customer_support_crew.crew import CustomerSupportCrew

def sanitize_filename(name_base: str, max_length: int = 60) -> str:
    """Sanitizes a string to be a valid filename component."""
    # Remove non-alphanumeric characters (except spaces, hyphens, underscores)
    name = re.sub(r'[^\w\s-]', '', name_base)
    # Replace whitespace and multiple hyphens/underscores with a single underscore
    name = re.sub(r'[-\s]+', '_', name).strip('_')
    # Truncate to max_length
    return name[:max_length]

def run():
    """
    Run the customer support crew.
    """
    # Define project root and output directory
    # Assuming this main.py is in src/customer_support_crew/
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_dir, "..", "..")) # Up two levels
    output_dir = os.path.join(project_root, "output")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Project root: {project_root}")
    print(f"Output directory: {output_dir}")

    # Define a sample customer query
    # Try queries like:
    # customer_query_input = "I need a refund for my order."
    # customer_query_input = "I can't log in to my account."
    # customer_query_input = "How long does shipping take for Product X?"
    # customer_query_input = "I need to cancel my subscription"
    customer_query_input = "My payment failed for my subscription, what should I do? It's urgent!"
    # customer_query_input = "Can you tell me how to be more empathetic with customers?"

    # Generate a unique filename stem for the output
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_query_part = sanitize_filename(customer_query_input)
    # This is the name that will be used in the {generated_filename} placeholder
    filename_stem = f"support_response_{timestamp}_{sanitized_query_part}"

    inputs = {
        'customer_query': customer_query_input,
        'generated_filename': filename_stem # Pass the filename stem
    }
    
    print(f"\nInput customer query: \"{customer_query_input}\"")
    print(f"Generated filename stem for output: \"{filename_stem}\"")
    
    # Create the crew
    support_crew_instance = CustomerSupportCrew()
    result = support_crew_instance.crew().kickoff(inputs=inputs)
    
    print("## Crew AI Kickoff Result (Raw final output):")
    # The 'result.raw' here is the string output of the last task in a sequential crew.
    # If the task's output_file is configured, CrewAI handles the saving.
    print(result.raw)

    # Confirm where the file was expected to be saved
    expected_file_path = os.path.join(output_dir, f"{filename_stem}.md")
    print(f"\nOutput Markdown file should be saved by CrewAI to: {expected_file_path}")

    if os.path.exists(expected_file_path):
        print(f"SUCCESS: Output file found at {expected_file_path}")
        # Optionally, print the content of the file
        # with open(expected_file_path, 'r', encoding='utf-8') as f:
        #     print("\n--- File Content ---")
        #     print(f.read())
        #     print("--- End of File Content ---")
    else:
        print(f"WARNING: Output file NOT found at {expected_file_path}.")
        print("Please check task configuration, agent's output format, and file permissions.")
        print("The agent's final output (result.raw) is printed above.")


if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set in your environment or .env file
    if not os.getenv("NVIDIA_NIM_API_KEY"):
        print("Error: NVIDIA_NIM_API_KEY environment variable not set.")
        print(f"Please set it in your .env file (expected at {dotenv_path}) or your environment.")
    else:
        run()