import pdfplumber

from crewai import Agent, Task, Crew
from crewai_tools import LlamaIndexTool
from langchain_ollama import ChatOllama
import os

from llama_index.core.tools import FunctionTool

from constants import LOCAL_EXPORT_DIR, PDF_DIR
from helpers.export_helper import ExportHelper
from helpers.models import Llama3_1, Model, MoonDream, Gemma2
from local.schemas import fielmann_schema, tarmed_schema

os.environ["OPENAI_API_KEY"] = "NA"


def run(model: Model, document_name: str, schema: dict):
    llm = ChatOllama(
        model=model.name,
        temperature=0.0,
        base_url="http://localhost:11434",
    )

    with pdfplumber.open(os.path.join(PDF_DIR, document_name)) as pdf:
        fielmann_text = pdf.pages[0].extract_text(layout=True)

    # def schema():
    #     return Invoice.model_json_schema()
    #
    # og_tool = FunctionTool.from_defaults(
    #     schema, name="<name>", description="<description>"
    # )
    # tool = LlamaIndexTool.from_tool(og_tool)

    extractor_agent = Agent(
        role="Information Extractor",
        goal="Extract relevant information from the supplied text into a json format.",
        backstory="You are an excellent information extractor that likes to extract information from given text into"
        "a structured format like json.",
        allow_delegation=False,
        verbose=True,
        llm=llm,
        # tools=[tool],
    )

    extraction_task = Task(
        description="Extract the relevant information defined in the expected output as json from the following "
        f"text: {fielmann_text}",
        agent=extractor_agent,
        expected_output=f"The extracted information with following structure: {schema}."
        "Make sure that each property has a valid value.",
    )

    reviewer_agent = Agent(
        role="Information Reviewer",
        goal="Review the extracted information from the supplied text and make sure it matches the supplied json "
        "format. If you find any mistakes, correct them.",
        backstory="You are an excellent information reviewer that likes to review extracted information from a given "
        "text into and makes sure it matches a structured format like json and ist a 100% correct.",
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    review_task = Task(
        description="Review the extracted information and make sure the format is correct. If you do corrections, do "
        "not annotate them in the output.",
        agent=reviewer_agent,
        expected_output=f"The extracted information with following structure: {schema}."
        "Make sure that each property has a valid value.",
    )

    crew = Crew(
        agents=[extractor_agent, reviewer_agent],
        tasks=[extraction_task, review_task],
        verbose=True,
    )

    result = crew.kickoff()

    print(result)

    ExportHelper().export_json_output(
        export_dir=LOCAL_EXPORT_DIR,
        document_name=document_name,
        model=model,
        output=result.tasks_output[0].raw,
    )


if __name__ == "__main__":
    # run(model=Llama3_1(), document_name="fielmann.pdf", schema=fielmann_schema)
    run(model=Llama3_1(), document_name="tarmed.pdf", schema=tarmed_schema)
    # run(model=Gemma2(), document_name="fielmann.pdf")
