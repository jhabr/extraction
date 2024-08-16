import json
import os
from typing import Optional

from openai.types.chat import ChatCompletion

from gpt4o.constants import GPT4o_EXPORT_DIR


class ExportHelper:

    def export_json_output(
        self,
        document_name: str,
        model_name: str,
        output: str,
        prefix: Optional[str] = None,
    ) -> None:
        with open(
            os.path.join(
                GPT4o_EXPORT_DIR,
                (
                    f"{prefix}_{document_name}_{model_name}.json"
                    if prefix
                    else f"{document_name}_{model_name}.json"
                ),
            ),
            "w",
        ) as json_file:
            json.dump(
                json.loads(output),
                json_file,
                indent=4,
            )

    def export_cost(
        self,
        model: dict,
        document_name: str,
        responses: [ChatCompletion],
        prefix: Optional[str] = None,
    ) -> None:
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        prompt_cost = 0
        completion_cost = 0
        total_cost = 0

        for response in responses:
            prompt_tokens += response.usage.prompt_tokens
            completion_tokens += response.usage.completion_tokens
            total_tokens += prompt_tokens + completion_tokens
            prompt_cost += model["input_cost"] * response.usage.prompt_tokens
            completion_cost += model["output_cost"] * response.usage.completion_tokens
            total_cost += prompt_cost + completion_cost

            with open(
                os.path.join(
                    GPT4o_EXPORT_DIR,
                    (
                        f"{prefix}_{document_name}_costs_{model['name']}.json"
                        if prefix
                        else f"{document_name}_costs_{model['name']}.json"
                    ),
                ),
                "w",
            ) as cost_json:
                json.dump(
                    {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                        "prompt_cost_$": round(prompt_cost, 6),
                        "completion_cost_$": round(completion_cost, 6),
                        "total_cost_$": round(prompt_cost + completion_cost, 6),
                    },
                    cost_json,
                    indent=4,
                )
