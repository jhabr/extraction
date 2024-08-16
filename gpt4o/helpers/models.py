class GPT4:
    name: str
    input_cost: float
    output_cost: float


class GPT4o(GPT4):
    name = "gpt-4o-2024-08-06"
    input_cost = 2.50 / 10e6
    output_cost = 10 / 10e6


class GPT4oMini(GPT4):
    name = "gpt-4o-mini-2024-07-18"
    input_cost = 0.150 / 10e6
    output_cost = 0.600 / 10e6


class GPT4Turbo(GPT4):
    name = "gpt-4-turbo"
    input_cost = 10.00 / 10e6
    output_cost = 30.00 / 10e6
