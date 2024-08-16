from pydantic import BaseModel, Field


class Customer(BaseModel):
    first_name: str = Field(description="the customer's first name")
    last_name: str = Field(description="the customer's last name")
    birthday: str = Field(description="the customer's birthday")
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
    identification: int = Field(description="the identification number of the invoice")
    positions: list[Position] = Field(
        description="all the positions with text and amount in the invoice"
    )
    total_price: float = Field(
        description="the total price of the invoice - sum of all listed positions"
    )


class Tarmed(BaseModel):
    name: str = Field(description="the name of the service provider")
    street: str = Field(description="the street name of the service provider")
    zip: int = Field(description="the zip code of the service provider")
    city: str = Field(description="the city name of the service provider")
    customer: Customer = Field(description="the customer structure in the invoice")
    invoice: Invoice = Field(description="the invoice structure")
