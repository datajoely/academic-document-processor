# Accademic journal processor

**Automated Extraction and Processing of Veterinary Research Documents**

## Project Description

The **Document Processor** is a Python-based tool designed to automate the extraction and processing of academic documents from various accademic journals. It supports multiple document formats, including PDF, DOCX, and HTML, and leverages language models to extract key information such as authors, titles, abstracts, and publication dates. This tool streamlines the organization and analysis of research papers, making it easier for researchers and professionals in the veterinary field to manage and utilize large volumes of academic literature.

## Features

- **Document Collection**: Automatically gathers documents from a structured directory of veterinary journals.
- **Text Extraction**: Extracts text from PDF, DOCX, and HTML files using reliable libraries.
- **Content Extraction**: Utilizes language models (e.g., GPT-4 or Llama3.2 (locally)) to extract structured information from document content.
- **Progress Monitoring**: Visual progress tracking using the `rich` library.
- **Error Handling**: Logs and manages both successful and failed document processing attempts.
- **Data Organization**: Structures extracted data into JSON Lines (`.jsonl`) format for easy access and analysis.

## Directory Structure

```
pdf-processor
├── .git
├── .ruff_cache
├── data
│   └── journals
│       ├── Acta Veterinaria Scandinavica
│       │   ├── 2017
│       │   ├── 2018
│       │   └── ...
│       ├── American Journal Veterinary Research
│       │   ├── 2016
│       │   ├── 2017
│       │   └── ...
│       └── ... (other journals)
├── extract_text.py
├── parse_documents.py
├── process_llm.py
└── print_utils.py
```

- **`data/journals/`**: Contains subdirectories for each journal, further organized by publication year and month ranges.
- **`extract_text.py`**: Handles text extraction from PDF, DOCX, and HTML files.
- **`parse_documents.py`**: Processes documents to extract structured information using defined models.
- **`process_llm.py`**: Interfaces with language models to extract specific content from the text.
- **`print_utils.py`**: Configures logging and console output using the `rich` library.

## Installation

## Usage

### Organize Your Data

Ensure your documents are placed in the `data/journals/{name}/{date}/{month_range}/{file}.{pdf|docx|htm|html}` directory following the existing structure. Each journal should have its own folder, subdivided by publication year and month ranges.

### Running the Document Parser

Execute the `parse_documents.py` script to process all documents and extract the desired information.

```bash
uv run parse_documents.py
```

This script will:

1. **Collect Documents**: Traverse the `data/journals/` directory to find all PDF, DOCX, and HTML files.
2. **Extract Text**: Use `extract_text.py` to extract text content from each document.
3. **Process Content**: Utilize `process_llm.py` to extract structured information such as authors, titles, abstracts, and publication dates using language models.
4. **Output Results**: Save successful extractions to `documents_success.jsonl` and any failures to `documents_failed.jsonl`.

### Monitoring Progress

The script provides a visual progress bar and logs information about the processing status using the `rich` library. Check the console output for real-time updates.

### Handling Outputs

- **documents_success.jsonl**: Contains JSON Lines with successfully extracted information.
- **documents_failed.jsonl**: Contains JSON Lines for documents that failed to process, along with error logs.

## Dependencies

The project relies on the following Python libraries:

- **pdfplumber**: For extracting text from PDF files.
- **beautifulsoup4**: For parsing HTML content.
- **python-docx**: For extracting text from DOCX files.
- **pydantic**: For data validation and settings management using Python type annotations.
- **rich**: For rich text and beautiful formatting in the terminal.
- **instructor**: Interface for language models (ensure proper configuration).
- **openai**: For interacting with OpenAI's or Ollama language models exposing an OpenAI dialect.
