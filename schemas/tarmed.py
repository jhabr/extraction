from pydantic import BaseModel, Field

from schemas.schemas import ServiceProvider, Customer


class Position(BaseModel):
    tariff_number: str = Field(
        description="the tariff number column of the position in the invoice"
    )
    text: str = Field(
        description="the textual description of the position in the invoice"
    )
    amount: float = Field(description="the quantity of the position in the invoice")
    price: float = Field(description="the price of the position in the invoice")


class Tarmed(BaseModel):
    date: str = Field(description="the date of the invoice")
    identification: int = Field(description="the identification number of the invoice")
    positions: list[Position] = Field(
        description="all the positions with text and amount in the invoice"
    )
    total_price: float = Field(
        description="the total price of the invoice - sum of all listed positions"
    )


class Invoice(BaseModel):
    service_provider: ServiceProvider
    customer: Customer
    invoice: Tarmed
