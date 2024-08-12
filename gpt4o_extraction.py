import base64
import os

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
    position: list[Position]
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

    response = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY")).chat.completions.create(
        model='gpt-4o-2024-08-06',
        messages=[
            {"role": "system", "content": "You are an assistant that extracts structured information from images."},
            {
                "role": "user",
                "content": (
                    "Analyze this image and provide the details in the structured format defined by the "
                    "Optician model.\n"
                    f"Image: {image_data}"  # Embed the image data directly in the message
                )
            },
        ],
        tools=[openai.pydantic_function_tool(Optician)],
    )

    # Print the structured output
    print(response)


if __name__ == "__main__":
    run()
