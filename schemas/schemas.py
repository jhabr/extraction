from pydantic import BaseModel, Field


class Customer(BaseModel):
    first_name: str = Field(description="the customer's first name")
    last_name: str = Field(description="the customer's last name")
    birthday: str = Field(description="the customer's birthday")
    street: str = Field(description="the customer's last name")
    zip: int = Field(description="the zip code of the city")
    city: str = Field(description="the city name")


class ServiceProvider(BaseModel):
    name: str = Field(description="the name of the service provider")
    street: str = Field(description="the street name of the service provider")
    zip: int = Field(description="the zip code of the service provider")
    city: str = Field(description="the city name of the service provider")
