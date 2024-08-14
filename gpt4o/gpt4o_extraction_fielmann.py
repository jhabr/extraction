import base64
import json
import os
import time

import openai
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from gpt4o.constants import GPT4o_EXPORT_DIR, GPT4o_IMAGES_DIR


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

    document_name = "fielmann@200"
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
    with open(os.path.join(GPT4o_IMAGES_DIR, f"{document_name}.jpg"), "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')

    start_time = time.time()

    print(f"Extracting {document_name} using {model['name']}...")

    response = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY")).chat.completions.create(
        model=model["name"],
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
        temperature=0,  # highest reproducibility
    )

    print(f"OpenAI response time: {time.time() - start_time:.2f} seconds")

    # Print the structured output
    print(response.choices[0].message.tool_calls[0].function.arguments)

    with open(os.path.join(GPT4o_EXPORT_DIR, f"{document_name}_{model['name']}.json"), "w") as json_file:
        json.dump(json.loads(response.choices[0].message.tool_calls[0].function.arguments), json_file, indent=4)

    prompt_cost = model["input_cost"] * response.usage.prompt_tokens
    completion_cost = model["output_cost"] * response.usage.completion_tokens

    with open(os.path.join(GPT4o_EXPORT_DIR, f"{document_name}_costs_{model['name']}.json"), "w") as cost_json:
        json.dump({
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "prompt_cost_$": round(prompt_cost, 6),
            "completion_cost_$": round(completion_cost, 6),
            "total_cost_$": round(prompt_cost + completion_cost, 6),
            "completion_time_s": round(time.time() - start_time, 2)
        }, cost_json, indent=4)


if __name__ == "__main__":
    run()
