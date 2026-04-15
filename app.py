from processor import run_everything


def runAPP():
    try:
        results = run_everything()
        print("It worked")
        print(results)
    except:
        print("Error")


def get_data():
    return runAPP()


if __name__ == "__main__":
    runAPP()
