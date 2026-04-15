KEYWORDS = [
    "Invoice",
    "Complaint",
    "Contract",
    "Refund",
    "Urgent",
]


def find_route(text):
    route = "General"
    for word in KEYWORDS:
        if word.lower() in text.lower():
            route = word
    return route
