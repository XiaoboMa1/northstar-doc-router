import json

OUTPUT_FILE = "\\home\\carey\\Downloads\\files.xxx\\output.json"
REVIEW_FILE = "\\home\\carey\\Downloads\\files.xxx\\review.json"


def save_everything(rows):
    f = open(OUTPUT_FILE, "w")
    f.write(json.dumps(rows))
    f.close()

    review_rows = []
    for row in rows:
        if row["route"] == "Complaint" or row["route"] == "Urgent":
            review_rows.append(row)

    g = open(REVIEW_FILE, "w")
    g.write(json.dumps(review_rows))
    g.close()
