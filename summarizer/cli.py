import argparse
from summarizer.model import summarize_batch, summarize_long_text


def main():
    parser = argparse.ArgumentParser(description="FLAN-T5 Text Summarizer")

    parser.add_argument("--text", type=str, help="Text to summarize")
    parser.add_argument("--file", type=str, help="Text file to summarize")
    parser.add_argument("--style", choices=["short", "medium"], default="medium")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--beams", type=int, default=4)

    args = parser.parse_args()

    # -------- Text input (single-pass) --------
    if args.text:
        summaries = summarize_batch(
            [args.text],
            style=args.style,
            temperature=args.temperature,
            num_beams=args.beams
        )

        print("\nSummary:\n", summaries[0])
        return

    # -------- File input (long-text pipeline) --------
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            raw_text = f.read()

        summary = summarize_long_text(
            raw_text,
            style=args.style,
            temperature=args.temperature,
            num_beams=args.beams
        )

        print("\nSummary:\n", summary)
        return

    print("Provide either --text or --file")


if __name__ == "__main__":
    main()
    