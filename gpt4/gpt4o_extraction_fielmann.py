import base64
import os
import time

import openai
from dotenv import load_dotenv

load_dotenv()
from openai import OpenAI

from constants import IMAGES_DIR
from helpers.export_helper import ExportHelper
from helpers.models import GPT4o, Model
from schemas.optician import Invoice


def run(document_name: str, model: Model):
    with open(os.path.join(IMAGES_DIR, f"{document_name}.jpg"), "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")

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
                        "text": "Analyze this image and provide the details in the structured format defined by the "
                        "Optician model.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpg;base64,{image_data}"},
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
        prefix="reviewed",
        document_name=document_name,
        responses=[response],
    )


if __name__ == "__main__":
    run(
        document_name="fielmann@200",
        model=GPT4o(),
    )
