"""Command-line equivalent of the second part of final.py (no Colab needed).

Usage:
    python run_analysis.py                 # analyses the bundled SRS.pdf
    python run_analysis.py path/to/doc.pdf --csv out.csv
"""
import argparse
import csv
import sys

from srs_classifier import SAMPLE_PDF_PATH, get_classifier


def main(argv=None):
    parser = argparse.ArgumentParser(description="Classify SRS PDF sentences with the saved model")
    parser.add_argument("pdf", nargs="?", default=SAMPLE_PDF_PATH)
    parser.add_argument("--csv", help="also write results to this CSV file")
    args = parser.parse_args(argv)

    results = get_classifier().analyze_pdf(args.pdf)

    print("Classification Results:\n")
    for r in results:
        print(f"Sentence: {r.sentence}")
        print(f"Predicted Label: {r.label}\n")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].to_dict()) if results else ["sentence"])
            writer.writeheader()
            for r in results:
                writer.writerow(r.to_dict())
        print(f"Saved {len(results)} rows to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
