import os

from file_store import save_everything
from llm_service import call_model
from metrics import counters, show_metrics
from routes import find_route


def run_everything():
    docs_path = "\\home\\carey\\Downloads\\files.xxx"
    all_results = []

    for file_name in os.listdir(docs_path):
        if file_name.endswith(".txt"):
            counters["seen"] = counters["seen"] + 1
            path = docs_path + "\\" + file_name
            f = open(path, "r")
            text = f.read()
            f.close()

            route = find_route(text)
            ai_result = call_model(text, route)

            row = {
                "file_name": file_name,
                "route": route,
                "llm": ai_result,
                "size": len(text),
            }
            all_results.append(row)
            counters["processed"] = counters["processed"] + 1

            if route == "Complaint" or route == "Urgent":
                counters["review"] = counters["review"] + 1

    save_everything(all_results)
    show_metrics()
    return all_results
