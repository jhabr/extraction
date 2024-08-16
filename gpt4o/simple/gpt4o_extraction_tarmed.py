import base64
import json
import os
import time

import openai
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from gpt4o.constants import GPT4o_IMAGES_DIR, GPT4o_EXPORT_DIR


class Customer(BaseModel):
    first_name: str = Field(description="the customer's first name")
    last_name: str = Field(description="the customer's last name")
    street: str = Field(description="the customer's last name")
    zip: int = Field(description="the zip code of the city")
    city: str = Field(description="the city name")


class Position(BaseModel):
    tariff_number: str = Field(
        description="the tariff number column of the position in the invoice"
    )
    text: str = Field(
        description="the textual description of the position in the invoice"
    )
    amount: float = Field(description="the quantity of the position in the invoice")
    price: float = Field(description="the price of the position in the invoice")


class Invoice(BaseModel):
    date: str = Field(description="the date of the invoice")
    invoice_number: int = Field(description="the identification number of the invoice")
    positions: list[Position] = Field(
        description="all the positions with text and amount in the invoice"
    )
    total_price: float = Field(
        description="the total price of the invoice - sum of all listed positions"
    )


class Doctor(BaseModel):
    name: str = Field(description="the name of the biller")
    street: str = Field(description="the street name of the biller")
    zip: int = Field(description="the zip code of the biller")
    city: str = Field(description="the city name of the biller")
    customer: Customer = Field(description="the customer structure in the invoice")
    invoice: Invoice = Field(description="the invoice structure")


def run():
    load_dotenv()

    document_name = "tarmed@200"
    model = {
        "name": "gpt-4o-2024-08-06",
        "input_cost": 2.50 / 10e6,
        "output_cost": 10 / 10e6,
    }
    # model = {
    #     "name": "gpt-4o-mini-2024-07-18",
    #     "input_cost": 0.150 / 10e6,
    #     "output_cost": 0.600 / 10e6,
    # }

    with open(
        os.path.join(GPT4o_IMAGES_DIR, f"{document_name}.jpg"), "rb"
    ) as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")

    start_time = time.time()

    print(f"Extracting {document_name} using {model['name']}...")

    response = OpenAI(api_key=os.getenv("OPENAI_API_KEY")).chat.completions.create(
        model=model["name"],
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
                        "Doctor model. Extract all positions in the provided image. Make sure extracted "
                        "information is correct.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpg;base64,{image_data}"},
                    },
                ],
            },
        ],
        tools=[openai.pydantic_function_tool(Doctor)],
        temperature=0,  # highest reproducibility
    )

    print(f"OpenAI response time: {time.time() - start_time:.2f} seconds")

    # Print the structured output
    print(response.choices[0].message.tool_calls[0].function.arguments)

    with open(
        os.path.join(GPT4o_EXPORT_DIR, f"{document_name}_{model['name']}.json"), "w"
    ) as json_file:
        json.dump(
            json.loads(response.choices[0].message.tool_calls[0].function.arguments),
            json_file,
            indent=4,
        )

    prompt_cost = model["input_cost"] * response.usage.prompt_tokens
    completion_cost = model["output_cost"] * response.usage.completion_tokens

    with open(
        os.path.join(GPT4o_EXPORT_DIR, f"{document_name}_costs_{model['name']}.json"),
        "w",
    ) as cost_json:
        json.dump(
            {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "prompt_cost_$": round(prompt_cost, 6),
                "completion_cost_$": round(completion_cost, 6),
                "total_cost_$": round(prompt_cost + completion_cost, 6),
                "completion_time_s": round(time.time() - start_time, 2),
            },
            cost_json,
            indent=4,
        )


if __name__ == "__main__":
    run()
