import sys


def main(): # TODO make choose model via list of models
    if len(sys.argv) < 2:
        print("Usage: engine <command>")
        return
    command = sys.argv[1]

    if command == "run":
        if len(sys.argv) < 3:
            print("Usage: engine run <model>")
            return
    
        model = sys.argv[2]
        print(f"Running model: {model}")
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()
