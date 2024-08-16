fielmann_schema = {
    "service_provider": {
        "name": str,
        "street": str,
        "zip": int,
        "city": str,
    },
    "customer": {
        "first_name": str,
        "last_name": str,
        "street": str,
        "zip": int,
        "city": str,
    },
    "invoice": {
        "date": str,
        "invoice_number": str,
        "positions": [
            {"text": str, "price": float},
            ...,
        ],
        "total_price": float,
    },
}

tarmed_schema = {
    "service_provider": {
        "name": str,
        "street": str,
        "zip": int,
        "city": str,
    },
    "customer": {
        "first_name": str,
        "last_name": str,
        "street": str,
        "zip": int,
        "city": str,
    },
    "invoice": {
        "date": str,
        "identification_number": str,
        "positions": [
            {"tariff_number": str, "text": str, "amount": float, "price": float},
            ...,
        ],
        "total_price": float,
    },
}
