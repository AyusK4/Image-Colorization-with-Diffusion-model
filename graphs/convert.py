import json
import os
import re
import sys


def load_metrics_from_txt(path: str):
    """
    Read a text file containing multiple JSON objects separated by one or more
    blank lines and return a list of parsed JSON objects.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split on one or more blank lines (handles Windows and Unix newlines)
    chunks = [c.strip() for c in re.split(r"\r?\n\s*\r?\n", content) if c.strip()]

    metrics = []
    for idx, chunk in enumerate(chunks, start=1):
        try:
            metrics.append(json.loads(chunk))
        except json.JSONDecodeError as e:
            print(f"Warning: skipping block {idx} due to JSON parse error: {e}", file=sys.stderr)

    return metrics


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "training Metrics.txt")
    output_path = os.path.join(base_dir, "training_metrics.json")

    metrics = load_metrics_from_txt(input_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote {len(metrics)} records to {os.path.basename(output_path)}")


if __name__ == "__main__":
    main()