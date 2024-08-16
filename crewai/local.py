from crewai import Agent, Task, Crew
from crewai_tools import LlamaIndexTool
from langchain_ollama import ChatOllama
import os

from llama_index.core.tools import FunctionTool

from constants import LOCAL_EXPORT_DIR
from helpers.export_helper import ExportHelper
from helpers.models import Llama3_1, Model, MoonDream
from schemas.optician import Invoice

os.environ["OPENAI_API_KEY"] = "NA"


def run(model: Model):
    llm = ChatOllama(model=model.name, base_url="http://localhost:11434")

    fielmann_text = """
            Fielmann AG, Bahnhofstrasse 32 6300 Zug
            Herr
            Hans Muster Anderstrasse 12 1234 Imdorf
            Rechnung : 12558 14564
            Für Hans Muster
            Wir liefern Ihnen gemäss ärtlicher Verordnung durhc Gemeinsch.-Praxis der Doktoren Augenzentrum Olten AG
            Fassung: Glas:
            Feme Rechts Feme Links
            Gesamtbetrag
            Fielmann MC 501 CL SL Durchmess 65, Kunststoff
            CHF A
            A A
            Betrag 19.90
            5.00 5.00
            29.90
            Sph
            + 2.00 + 2.00
            Cyl Achse 0.75 99 -0.75 85
            Prisma
            Basis
            Add
            A: 7.7 % MwSt. in 29.90 CHF entsprechen 2.15 CHF. Netto Betrag 27.75 CHF.
            Zug, den 19.12.21 Ihr Auftrag 12558 14564 vom 12.12.2021
            NIF 49585
        """

    # def schema():
    #     return Invoice.model_json_schema()
    #
    # og_tool = FunctionTool.from_defaults(
    #     schema, name="<name>", description="<description>"
    # )
    # tool = LlamaIndexTool.from_tool(og_tool)

    general_agent = Agent(
        role="Information Extractor",
        goal="Extract relevant information from the supplied text into a json format.",
        backstory="You are an excellent information extractor that likes to extract information from given text into"
        "a structured format like json.",
        allow_delegation=False,
        verbose=True,
        llm=llm,
        # tools=[tool],
    )

    schema = {
        "name": str,
        "street": str,
        "zip": int,
        "city": str,
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

    task = Task(
        description=f"Extract the relevant information defined in the expected output as json from the following "
        f"text: {fielmann_text}",
        agent=general_agent,
        expected_output=f"The extracted information with following structure: {schema}."
        f"Make sure that each property has a value.",
    )

    crew = Crew(agents=[general_agent], tasks=[task], verbose=True)

    result = crew.kickoff()

    print(result)

    ExportHelper().export_json_output(
        export_dir=LOCAL_EXPORT_DIR,
        document_name="fielmann",
        model=model,
        output=result.tasks_output[0].raw,
    )


if __name__ == "__main__":
    run(model=Llama3_1())
