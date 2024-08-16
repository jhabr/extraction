import base64
import os
import time

from dotenv import load_dotenv

load_dotenv()

import openai
from openai import OpenAI

from gpt4o.constants import GPT4o_IMAGES_DIR
from gpt4o.helpers.export_helper import ExportHelper
from gpt4o.schemas.schemas import Tarmed


def run():
    document_name = "tarmed@200"
    model = {
        "name": "gpt-4o-2024-08-06",
        "input_cost": 2.50 / 10e6,
        "output_cost": 10 / 10e6,
    }

    with open(
        os.path.join(GPT4o_IMAGES_DIR, f"{document_name}.jpg"), "rb"
    ) as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")

    start_time = time.time()

    print(f"Extracting {document_name} using {model['name']}...")

    extraction_response = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    ).chat.completions.create(
        model=model["name"],
        messages=[
            {
                "role": "system",
                "content": "You are an assistant that extracts structured information from images. "
                "Use the supplied tools to assist the user.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this image and provide the details in the structured format "
                        "defined by the Tarmed model tool. Extract all positions in the provided image. "
                        "Make sure extracted information is correct.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpg;base64,{image_data}"},
                    },
                ],
            },
        ],
        tools=[openai.pydantic_function_tool(Tarmed)],
        temperature=0,
    )

    extracted_data = (
        extraction_response.choices[0].message.tool_calls[0].function.arguments
    )

    print(f"Reviewing {document_name} using {model['name']}...")

    review_response = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    ).chat.completions.create(
        model=model["name"],
        messages=[
            {
                "role": "system",
                "content": "You are an assistant that reviews structured information extracted from images. "
                "Verify and correct the information.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Review the following information for accuracy and correctness and make sure that "
                            f"the extracted data is a 100% correct: {extracted_data}"
                            "If the extracted information is not correct, fix the information by extracting it "
                            "from the supplied image. Use the provided Tarmed model tool for output structure."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpg;base64,{image_data}"},
                    },
                ],
            },
        ],
        tools=[openai.pydantic_function_tool(Tarmed)],
        temperature=0,
    )

    print(f"OpenAI response time: {time.time() - start_time:.2f} seconds")

    export_helper = ExportHelper()

    reviewed_data = review_response.choices[0].message.tool_calls[0].function.arguments

    export_helper.export_json_output(
        prefix="reviewed",
        document_name=document_name,
        model_name=model["name"],
        output=reviewed_data,
    )
    export_helper.export_cost(
        model=model,
        prefix="reviewed",
        document_name=document_name,
        responses=[extraction_response, review_response],
    )

    print(reviewed_data)


if __name__ == "__main__":
    run()
