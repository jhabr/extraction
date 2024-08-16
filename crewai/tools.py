import os

import openai
from crewai_tools import BaseTool
from openai import OpenAI

from schemas.schemas import Tarmed


class ExtractionTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="Extraction Tool",
            description="Tool to extract document information",
        )

    def _run(self, params: dict) -> str:
        response = OpenAI(api_key=os.getenv("OPENAI_API_KEY")).chat.completions.create(
            model="gpt-4",
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
                            "defined by the Doctor model. Extract all positions in the provided image. "
                            "Make sure extracted information is correct.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpg;base64,{params['image_data']}"
                            },
                        },
                    ],
                },
            ],
            tools=[openai.pydantic_function_tool(Tarmed)],
            temperature=0,
        )

        extracted_data = response.choices[0].message.tool_calls[0].function.arguments
        return extracted_data


class ReviewTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="Reviewing Tool",
            description="Tool to review extracted document information",
        )

    def _run(self, params: dict) -> str:
        response = OpenAI(api_key=os.getenv("OPENAI_API_KEY")).chat.completions.create(
            model="gpt-4",
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
                            "text": "Review the following information for accuracy and correctness: "
                            f"{params['extracted_data']}",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpg;base64,{params['image_data']}"
                            },
                        },
                    ],
                },
            ],
            temperature=0,
        )

        reviewed_data = response.choices[0].message.tool_calls[0].function.arguments
        return reviewed_data
