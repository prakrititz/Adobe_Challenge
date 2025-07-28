# Adobe India Hackathon: Connecting the Dots

## Table of Contents
- [Overview](#overview)
- [Round 1A: Universal Document Structure Extractor](#round-1a-universal-document-structure-extractor)
- [Round 1B: Persona-Driven Document Intelligence](#round-1b-persona-driven-document-intelligence)
- [Installation & Setup](#installation--setup)
- [Usage Instructions](#usage-instructions)
- [Technical Specifications](#technical-specifications)
- [Troubleshooting](#troubleshooting)

## Overview

This repository contains our complete submission for the *Adobe India Hackathon: Connecting the Dots*. Our project demonstrates a journey through two challenging rounds:

- **Round 1A**: Universal document structure parser
- **Round 1B**: Sophisticated, persona-driven intelligence engine

Our core philosophy focuses on building robust, efficient, and offline-first solutions that handle diverse documents and user needs under strict performance constraints.

---

## Round 1A: Universal Document Structure Extractor

### Challenge
Build a system that ingests any PDF and automatically extracts a structured outline, including the document's title and hierarchy of headings (H1, H2, H3). The solution must be universal, relying on structural and visual cues rather than language-specific keywords.

### Solution Approach
Multi-stage analysis pipeline that decodes a document's visual language:

![Round 1A Solution Architecture](1a_solution/git_1a.png)

1. **Noise Cancellation**: Using PyMuPDF to identify and strip irrelevant content (tables, headers, footers)
2. **Typographic Analysis**: Statistical analysis of font styles to identify primary "body text" style
3. **Visual Hierarchy Mapping**: Deviation from body style indicates potential headings, mapped to logical hierarchy using font size, weight, and indentation
4. **Language Detection**: fasttext integration for multilingual document support

### Technology Stack
- **Core**: Python 3
- **PDF Parsing**: PyMuPDF==1.23.14
- **Language Detection**: fasttext with lid.176.ftz model
- **Model Size**: < 200MB
- **Utilities**: requests, tqdm

### Docker Setup & Usage

#### Prerequisites
- Docker installed
- Input directory with PDF files
- Output directory for JSON files

#### Building the Image
```bash
docker build --platform linux/amd64 -t pdf-extractor:latest .
```

#### Running the Application
1. Create directories:
```bash
mkdir -p input output
```

2. Place PDF files in `input` directory

3. Run container:
```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none pdf-extractor:latest
```

#### Expected Output
- For each `document.pdf`, generates `document.json` in output directory
- Structured JSON with title and heading hierarchy

---

## Round 1B: Persona-Driven Document Intelligence

### Challenge
Build an intelligent analyst that processes document collections and extracts relevant sections based on user persona and "job-to-be-done". Must operate CPU-only, offline, with strict memory constraints (< 1GB model size).

### Solution: The Verification Gauntlet
Multi-stage verification approach optimized for small, offline models:

![Round 1B Solution Architecture](1b_solution/git_1b.png)

1. **Broad Semantic Search**: all-MiniLM-L6-v2 embeddings for initial relevance pool
2. **Keyword Elimination**: Fast filtering for hard constraints (e.g., dietary restrictions)
3. **LLM Verification**: reasoning-llama3.2-1b model for binary relevance verification
4. **Summarization**: Concise summaries and human-readable section titles

### Technology Stack
- **Core**: Python 3
- **Embeddings**: sentence-transformers==2.2.2 (all-MiniLM-L6-v2)
- **LLM**: llama-cpp-python==0.2.11 (reasoning-llama3.2-1b.Q6_K.gguf)
- **Model Size**: < 1GB
- **ML Libraries**: numpy==1.24.3, torch==2.0.1, scikit-learn==1.3.0

### Docker Setup & Usage

#### Project Structure Required
```
your-project-directory/
├── Collection 1/
│   ├── challenge1b_input.json
│   ├── PDFs/
│   │   ├── document1.pdf
│   │   └── document2.pdf
│   └── challenge1b_output.json (generated)
├── Collection 2/
│   └── ... (same structure)
```

#### Input Format
```json
{
  "documents": [
    {"filename": "document1.pdf"},
    {"filename": "document2.pdf"}
  ],
  "persona": {
    "role": "Data Scientist"
  },
  "job_to_be_done": {
    "task": "Find machine learning techniques for data analysis"
  }
}
```

#### Building & Running
```bash
# Build image
docker build -t pdf-analyzer:latest .

# Run analysis
docker run --rm -v $(pwd):/app --network none pdf-analyzer:latest
```

#### Output Format
```json
{
  "metadata": {
    "input_documents": ["document1.pdf", "document2.pdf"],
    "persona": "Data Scientist",
    "job_to_be_done": "Find machine learning techniques for data analysis",
    "processing_timestamp": "2025-07-28T10:30:00"
  },
  "extracted_sections": [
    {
      "document": "document1.pdf",
      "section_title": "Machine Learning Algorithms Overview",
      "importance_rank": 1,
      "page_number": 3
    }
  ],
  "subsection_analysis": [
    {
      "document": "document1.pdf",
      "refined_text": "Detailed text content from the relevant section...",
      "page_number": 3
    }
  ]
}
```

---

## Installation & Setup

### Prerequisites
- Python 3.9+
- Docker (for containerized deployment)
- Virtual environment (recommended)

### Local Development Setup
1. Clone repository:
```bash
git clone https://github.com/prakrititz/Adobe_Challenge.git
cd Adobe_Challenge
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage Instructions

### Round 1A Usage
1. Place PDF files in `input/` directory
2. Run Docker container as described above
3. Check `output/` directory for generated JSON files

### Round 1B Usage
1. Organize files in Collection folders structure
2. Create `challenge1b_input.json` files with persona and task definitions
3. Run Docker container
4. Review generated `challenge1b_output.json` and debug logs

---

## Technical Specifications

### Performance Constraints
- **Round 1A**: Model size < 200MB
- **Round 1B**: Model size < 1GB, CPU-only operation
- **Both**: Offline operation, no network dependencies

### Supported Platforms
- Linux (primary)
- Windows (with PowerShell/CMD adjustments)
- macOS

### Dependencies Overview
- **PDF Processing**: PyMuPDF, pdfplumber
- **ML/AI**: sentence-transformers, llama-cpp-python, torch
- **Language Detection**: fasttext
- **Utilities**: numpy, scikit-learn, requests, tqdm

---

## Troubleshooting

### Common Issues

#### Round 1A
- **No output files**: Check input directory contains valid PDF files
- **Permission errors**: Adjust directory permissions (`chmod -R 755`)

#### Round 1B
- **No Collection folders found**: Ensure folders named "Collection 1", "Collection 2", etc.
- **Out of memory**: Increase Docker memory limit for large PDF collections
- **Missing PDFs**: Verify PDFs exist in `PDFs/` subfolder and match input JSON references

### Debugging
- Check container logs: `docker logs <container-id>`
- Review debug files: `all_findings_with_scores.txt` in Collection folders
- Verify input JSON format matches expected schema

### Platform-Specific Commands

#### Windows PowerShell
```powershell
docker run --rm -v ${PWD}:/app --network none pdf-analyzer:latest
```

#### Windows Command Prompt
```cmd
docker run --rm -v %cd%:/app --network none pdf-analyzer:latest
```

---

## Performance Notes
- Processing time scales with PDF size and document count
- Embedding generation is most time-intensive step
- Large collections may require several minutes
- CPU-only inference optimized for resource constraints
