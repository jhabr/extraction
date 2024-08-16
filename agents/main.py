import base64
import os
import time

from dotenv import load_dotenv

from helpers.models import GPT4o, Model
from schemas.tarmed import Tarmed

load_dotenv()

import openai
from openai import OpenAI

from constants import IMAGES_DIR
from helpers.export_helper import ExportHelper


def run(
    document_name: str,
    model: Model,
    reviews: int = 1,
):
    responses = []

    with open(os.path.join(IMAGES_DIR, f"{document_name}.jpg"), "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")

    start_time = time.time()

    print(f"Extracting {document_name} using {model.name}...")

    extraction_response = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    ).chat.completions.create(
        model=model.name,
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

    responses.append(extraction_response)

    review_data = (
        extraction_response.choices[0].message.tool_calls[0].function.arguments
    )

    for index in range(reviews):
        print(f"{index + 1}. Review of {document_name} using {model.name}...")

        review_response = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        ).chat.completions.create(
            model=model.name,
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
                                "the extracted data is a 100% correct. If the extracted information is not correct, "
                                "fix the information by extracting it from the supplied image."
                                "Use the provided Tarmed model tool for output structure."
                            ),
                        },
                        {
                            "type": "text",
                            "text": f"Here is the extracted data in json format: {review_data}",
                        },
                        {
                            "type": "text",
                            "text": "Here is the image data that needs to be analyzed.",
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

        responses.append(review_response)

        review_data = (
            review_response.choices[0].message.tool_calls[0].function.arguments
        )

    print(f"OpenAI response time: {time.time() - start_time:.2f} seconds")

    export_helper = ExportHelper()

    export_helper.export_json_output(
        prefix="reviewed",
        document_name=document_name,
        model=model,
        output=review_data,
    )
    export_helper.export_cost(
        model=model,
        prefix="reviewed",
        document_name=document_name,
        responses=responses,
    )

    print(review_data)


if __name__ == "__main__":
    run(
        document_name="tarmed@200",
        model=GPT4o(),
        reviews=2,
    )
