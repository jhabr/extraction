import base64
import os
import time

import openai
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


class Customer(BaseModel):
    name: str
    street: str
    zip: int
    city: str


class Position(BaseModel):
    text: str
    price: float


class Invoice(BaseModel):
    date: str
    invoice_number: int
    positions: list[Position]
    total_price: float


class Optician(BaseModel):
    name: str
    street: str
    zip: int
    city: str
    customer: Customer
    invoice: Invoice


def run():
    load_dotenv()

    with open("/Users/jh/Downloads/documents/fielmann.jpg", "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')

    start_time = time.time()

    response = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY")).chat.completions.create(
        model='gpt-4o-2024-08-06',
        messages=[
            {
                "role": "system",
                "content": "You are an assistant that extracts structured information from images."
                           "Use the supplied tools to assist the user."},
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
                        "image_url": {
                            "url": f"data:image/jpg;base64,{image_data}"
                        }
                    }
                ]
            },
        ],
        tools=[openai.pydantic_function_tool(Optician)],
        temperature=0,
    )

    # Print the structured output
    print(response.choices[0].message.tool_calls[0].function.arguments)

    print(f"OpenAI response time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    run()
