from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# Import the custom tool
from .tools.conversation_query_tool import ConversationQueryTool

@CrewBase
class CustomerSupportCrew:
    """CustomerSupportCrew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self):
        # Initialize the tool.
        # The path to the dataset is relative to the project root.
        # If crew.py is in src/customer_support_crew/, data/ is ../../data/
        self.conversation_query_tool = ConversationQueryTool(dataset_path="data/sample_conversations.json")

    @agent
    def support_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['support_agent'],
            tools=[self.conversation_query_tool], # Assign the tool to the agent
            verbose=True
        )

    @task
    def handle_customer_query_task(self) -> Task:
        return Task(
            config=self.tasks_config['handle_customer_query'],
            agent=self.support_agent() # Assign the agent to the task
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Customer Support crew"""
        return Crew(
            agents=[self.support_agent()],
            tasks=[self.handle_customer_query_task()],
            process=Process.sequential,
            verbose=2 # You can set it to 1 or 2 for different levels of verbosity
        )