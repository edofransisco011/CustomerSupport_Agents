import os
from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file

from customer_support_crew.crew import CustomerSupportCrew

def run():
    """
    Run the customer support crew.
    """
    # Define a sample customer query
    # Try queries like:
    # "I need a refund for my order."
    # "I can't log in to my account."
    # "How long does shipping take for Product X?"
    # "I need to cancel my subscription"
    # "My payment failed, what should I do?" (This one won't have direct matches, let's see how the agent handles it)
    customer_query_input = "I'm having trouble with my credit card payment for my subscription, it keeps getting declined. What should I do?"

    inputs = {
        'customer_query': customer_query_input
    }
    
    # Create the crew
    support_crew_instance = CustomerSupportCrew()
    result = support_crew_instance.crew().kickoff(inputs=inputs)
    
    print("Customer Agent Response:")
    print(result)

if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set in your environment or .env file
    if not os.getenv("NVIDIA_NIM_API_KEY"):
        print("Error: NVIDIA_NIM_API_KEY environment variable not set.")
        print("Please set it in your .env file or your environment.")
    else:
        run()