class Model:
    name: str
    input_cost: float
    output_cost: float


class GPT4o(Model):
    name = "gpt-4o-2024-08-06"
    input_cost = 2.50 / 10e6
    output_cost = 10 / 10e6


class GPT4oMini(Model):
    name = "gpt-4o-mini-2024-07-18"
    input_cost = 0.150 / 10e6
    output_cost = 0.600 / 10e6


class GPT4Turbo(Model):
    name = "gpt-4-turbo"
    input_cost = 10.00 / 10e6
    output_cost = 30.00 / 10e6


class Llama3_1(Model):
    name = "llama3.1"
    input_cost = 0.0
    output_cost = 0.0


class Gemma2(Model):
    name = "gemma2:2b"
    input_cost = 0.0
    output_cost = 0.0


class MoonDream(Model):
    name = "moondream"
    input_cost = 0.0
    output_cost = 0.0
