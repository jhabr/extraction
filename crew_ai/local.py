import pdfplumber

from crewai import Agent, Task, Crew
from crewai_tools import LlamaIndexTool
from langchain_ollama import ChatOllama
import os

from llama_index.core.tools import FunctionTool

from constants import LOCAL_EXPORT_DIR, PDF_DIR
from helpers.export_helper import ExportHelper
from helpers.models import Llama3_1, Model, MoonDream
from schemas.optician import Invoice

os.environ["OPENAI_API_KEY"] = "NA"


def run(model: Model, document_name: str):
    llm = ChatOllama(model=model.name, base_url="http://localhost:11434")

    with pdfplumber.open(os.path.join(PDF_DIR, document_name)) as pdf:
        fielmann_text = pdf.pages[0].extract_text(layout=True)

    # def schema():
    #     return Invoice.model_json_schema()
    #
    # og_tool = FunctionTool.from_defaults(
    #     schema, name="<name>", description="<description>"
    # )
    # tool = LlamaIndexTool.from_tool(og_tool)

    schema = {
        "service_provider": {
            "name": str,
            "street": str,
            "zip": int,
            "city": str,
        },
        "customer": {
            "first_name": str,
            "last_name": str,
            "street": str,
            "zip": int,
            "city": str,
        },
        "invoice": {
            "date": str,
            "invoice_number": str,
            "positions": [
                {"text": str, "price": float},
                ...,
            ],
            "total_price": float,
        },
    }

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
        description=f"Extract the relevant information defined in the expected output as json from the following "
        f"text: {fielmann_text}",
        agent=extractor_agent,
        expected_output=f"The extracted information with following structure: {schema}."
        f"Make sure that each property has a value.",
    )

    crew = Crew(
        agents=[extractor_agent],
        tasks=[extraction_task],
        verbose=True,
    )

    result = crew.kickoff()

    print(result)

    ExportHelper().export_json_output(
        export_dir=LOCAL_EXPORT_DIR,
        document_name="fielmann",
        model=model,
        output=result.tasks_output[0].raw,
    )


if __name__ == "__main__":
    run(model=Llama3_1(), document_name="fielmann.pdf")
