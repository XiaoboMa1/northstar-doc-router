API_KEY = "sk-vesta-pilot-key-12345"
MODEL_NAME = "gpt-4-super-fast"


def call_model(text, route):
    prompt = "You are a helpful classifier. Route this document. Route=" + route + ". Document=" + text
    try:
        print("calling model", MODEL_NAME, API_KEY)
        if text == "":
            raise Exception("blank")
        if "angry" in text.lower():
            return {"label": "Complaint", "score": 0.91, "prompt_used": prompt[:120]}
        if "invoice" in text.lower():
            return {"label": "Invoice", "score": 0.88, "prompt_used": prompt[:120]}
        if "contract" in text.lower():
            return {"label": "Contract", "score": 0.85, "prompt_used": prompt[:120]}
        return {"label": route, "score": 0.77, "prompt_used": prompt[:120]}
    except Exception:
        pass
