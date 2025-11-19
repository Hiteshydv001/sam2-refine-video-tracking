import argparse
import csv
import os
from typing import Dict, List


def load_metrics(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def summarize(run_dir: str) -> Dict[str, float]:
    csv_path = os.path.join(run_dir, "metrics", "run_metrics.csv")
    rows = load_metrics(csv_path)
    def collect(field: str) -> List[float]:
        return [float(row[field]) for row in rows if row[field]]
    reliable_ratio = sum(1 for row in rows if row.get("status", "").startswith("Tracking")) / len(rows)
    return {
        "mean_iou": mean(collect("iou")),
        "mean_centroid_error": mean(collect("centroid_error")),
        "mean_quality": mean(collect("quality")),
        "mean_threshold": mean(collect("threshold")),
        "tracking_ratio": reliable_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize tracking metrics across runs")
    parser.add_argument(
        "runs",
        nargs="+",
        help="One or more artifact run directories (e.g., artifacts/proposed)",
    )
    args = parser.parse_args()
    for run in args.runs:
        stats = summarize(run)
        print(run)
        for key, value in stats.items():
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
