from crewai import Agent, Task, Crew
from langchain_ollama import ChatOllama
import os

from gpt4o.constants import LOCAL_EXPORT_DIR
from gpt4o.helpers.export_helper import ExportHelper
from gpt4o.helpers.models import LLAMA3_1

os.environ["OPENAI_API_KEY"] = "NA"


def run():
    llm = ChatOllama(model="llama3.1", base_url="http://localhost:11434")

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

    output = {
        "customer": {"first_name": "extracted_value", "last_name": "extracted_value"}
    }

    general_agent = Agent(
        role="Information Extractor",
        goal="Extract relevant information from the supplied text into a json format.",
        backstory="You are an excellent information extractor that likes to extract information from given text into"
        "a structured format like json.",
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    task = Task(
        description=f"Extract the relevant information defined in the expected output as json from the following "
        f"text: {fielmann_text}",
        agent=general_agent,
        expected_output=f"fThe extracted information in json format with the following structure: {output}",
    )

    crew = Crew(agents=[general_agent], tasks=[task], verbose=True)

    result = crew.kickoff()

    print(result)

    ExportHelper().export_json_output(
        export_dir=LOCAL_EXPORT_DIR,
        document_name="fielmann",
        model=LLAMA3_1(),
        output=result.tasks_output[0].raw,
    )


if __name__ == "__main__":
    run()
