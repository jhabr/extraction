import base64
import os

from crewai import Crew, Task, Process
from dotenv import load_dotenv

from gpt4o.constants import IMAGES_DIR

load_dotenv()

from gpt4o.crewai.agents import ExtractorAgent, ReviewerAgent
from gpt4o.crewai.tools import ExtractionTool, ReviewTool


def run():
    load_dotenv()

    with open(os.path.join(IMAGES_DIR, "fielmann@200.jpg"), "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")

    extractor_agent = ExtractorAgent()
    reviewer_agent = ReviewerAgent()

    extraction_task = Task(
        description="Extract information from document",
        expected_output="A string in json format that contains all information in the document",
        agent=extractor_agent,
        tools=[ExtractionTool()],
    )

    review_task = Task(
        description="Review the extracted information",
        expected_output="A string in json format that contains all information in the document which is a 100% correct",
        agent=reviewer_agent,
        tools=[ReviewTool()],
    )

    crew = Crew(
        agents=[extractor_agent, reviewer_agent],
        tasks=[extraction_task, review_task],
        full_output=True,
        verbose=True,
    )

    result = crew.kickoff(inputs={"image_data": image_data})
    print(result)


if __name__ == "__main__":
    run()
