import os
import time

import openai
from dotenv import load_dotenv

from helpers.export_helper import ExportHelper
from schemas.optician import Invoice

load_dotenv()
from openai import OpenAI

from helpers.models import Model, GPT4Turbo


def run(
    document_name: str,
    model: Model,
):
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

    start_time = time.time()

    print(f"Extracting {document_name} using {model.name}...")

    response = OpenAI(api_key=os.getenv("OPENAI_API_KEY")).chat.completions.create(
        model=model.name,
        messages=[
            {
                "role": "system",
                "content": "You are an assistant that extracts structured information from images."
                "Use the supplied tools to assist the user.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze and return the details in the structured format defined "
                        f"by the Optician model from the following document: {fielmann_text}",
                    },
                ],
            },
        ],
        tools=[openai.pydantic_function_tool(Invoice)],
        temperature=0,  # highest reproducibility
    )

    print(f"OpenAI response time: {time.time() - start_time:.2f} seconds")

    # Print the structured output
    output = response.choices[0].message.tool_calls[0].function.arguments
    print(output)

    export_helper = ExportHelper()

    export_helper.export_json_output(
        document_name=document_name,
        model=model,
        output=output,
    )
    export_helper.export_cost(
        model=model,
        document_name=document_name,
        responses=[response],
    )


if __name__ == "__main__":
    run(
        document_name="fielmann@200",
        model=GPT4Turbo(),
    )
