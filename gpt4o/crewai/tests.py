from crewai import Crew, Agent, Task, Process
from dotenv import load_dotenv

load_dotenv()
from langchain_community.tools import DuckDuckGoSearchRun
from crewai_tools import tool


@tool("DuckDuckGoSearch")
def search(search_query: str):
    """Search the web for information on a given topic"""
    return DuckDuckGoSearchRun().run(search_query)


# Define crewai with specific roles and tools
researcher = Agent(
    role="Senior Research Analyst",
    goal="Discover innovative AI technologies",
    backstory="""You're a senior research analyst at a large company.
        You're responsible for analyzing data and providing insights
        to the business.
        You're currently working on a project to analyze the
        trends and innovations in the space of artificial intelligence.""",
    tools=[search],
)

writer = Agent(
    role="Content Writer",
    goal="Write engaging articles on AI discoveries",
    backstory="""You're a senior writer at a large company.
        You're responsible for creating content to the business.
        You're currently working on a project to write about trends
        and innovations in the space of AI for your next meeting.""",
    verbose=True,
)

# Create tasks for the crewai
research_task = Task(
    description="Identify breakthrough AI technologies",
    agent=researcher,
    expected_output="A bullet list summary of the top 5 most important AI news",
)
write_article_task = Task(
    description="Draft an article on the latest AI technologies",
    agent=writer,
    expected_output="3 paragraph blog post on the latest AI technologies",
)

# Assemble the crew with a sequential process
my_crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_article_task],
    process=Process.sequential,
    full_output=True,
    verbose=True,
)

if __name__ == "__main__":
    result = my_crew.kickoff()
    print(result)
