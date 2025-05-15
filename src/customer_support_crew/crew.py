from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
import os
import logging

# Import the custom tool
from .tools.conversation_query_tool import ConversationQueryTool

# Configure logging
logger = logging.getLogger(__name__)

@CrewBase
class CustomerSupportCrew:
    """CustomerSupportCrew for handling customer support inquiries"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self, dataset_path="data/sample_conversations.json", llm_model=None, llm_provider=None):
        """
        Initialize the CustomerSupportCrew.
        
        Args:
            dataset_path (str): Path to the conversation dataset JSON file
            llm_model (str, optional): The LLM model to use (overrides config)
            llm_provider (str, optional): The LLM provider to use (overrides config)
        """
        # Resolve the dataset path
        try:
            # Attempt to make path absolute if it's not already
            if not os.path.isabs(dataset_path):
                base_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.abspath(os.path.join(base_dir, '..', '..'))
                dataset_path = os.path.join(project_root, dataset_path)
                
            logger.info(f"Using conversation dataset at: {dataset_path}")
            
            # Initialize the tool with the dataset path
            self.conversation_query_tool = ConversationQueryTool(dataset_path=dataset_path)
            
            # Store the LLM configuration for later use
            self.llm_override = None
            if llm_provider and llm_model:
                self.llm_override = f"{llm_provider}/{llm_model}"
                logger.info(f"Using custom LLM: {self.llm_override}")
            elif llm_model:
                self.llm_override = llm_model
                logger.info(f"Using custom LLM model: {self.llm_override}")
                
        except Exception as e:
            logger.error(f"Error initializing CustomerSupportCrew: {e}")
            # Initialize with empty dataset as fallback
            self.conversation_query_tool = ConversationQueryTool()
            self.llm_override = None

    @agent
    def support_agent(self) -> Agent:
        """Create the customer support agent with tools and configuration"""
        # Get the base configuration
        agent_config = self.agents_config['support_agent']
        
        # Override LLM if specified
        if self.llm_override:
            agent_config['llm'] = self.llm_override
            
        return Agent(
            config=agent_config,
            tools=[self.conversation_query_tool],
            verbose=True
        )

    @task
    def handle_customer_query_task(self) -> Task:
        """Create the task for handling customer queries"""
        return Task(
            config=self.tasks_config['handle_customer_query'],
            agent=self.support_agent()
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Customer Support crew with the configured agent and task"""
        return Crew(
            agents=[self.support_agent()],
            tasks=[self.handle_customer_query_task()],
            process=Process.sequential,
            verbose=True
        )