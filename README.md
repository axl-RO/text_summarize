# FLAN-T5 Text Summarizer (CLI)

A lightweight **command-line text summarization tool** built using **FLAN-T5**.
It supports short and medium summaries and can handle **large `.txt` files** through automatic chunking.

## Features

- FLAN-T5 (`google/flan-t5-small`)
- CLI-based usage
- Supports raw text and `.txt` files
- Short & medium summary styles
- Handles long text via hierarchical chunking
- Installable as a global CLI command

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Summarize text

```bash
t5summarize --text "Your text here"
```

### Summarize a file

```bash
t5summarize --file sample.txt
```

### Choose style

```bash
t5summarize --file sample.txt --style short
```

## Model

- **Model**: FLAN-T5 (encoder–decoder transformer)
- **Long text handling**: Chunking + summary-of-summaries
