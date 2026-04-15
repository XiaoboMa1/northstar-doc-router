counters = {
    "seen": 0,
    "processed": 0,
    "review": 0,
}


def show_metrics():
    print("docs seen", counters["seen"])
    print("docs processed", counters["processed"])
    print("docs for review", counters["review"])
