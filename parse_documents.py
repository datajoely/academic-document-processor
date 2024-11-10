from __future__ import annotations

import json
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

from pydantic import BaseModel, computed_field
from pydantic_core import ValidationError
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

import extract_text
from print_utils import CONSOLE, LOGGER
from process_llm import ContentExtractor


class ResearchPaperSummary(BaseModel):
    """Model to hold research paper summary including authors, title, and abstract."""

    authors: list[str] | None
    title: str
    abstract: str


class ResearchPaperDates(BaseModel):
    """Model to hold the start and end dates for a research paper."""

    start_date: date
    end_date: date


class Document(BaseModel):
    """Represents a document to be processed with metadata such as journal, year, and path."""

    journal: str
    year: int | None
    month_range: str | None
    path: pathlib.Path

    @computed_field
    def kind(self) -> str:
        """Returns the file extension type of the document, e.g., PDF, DOCX."""
        return self.path.suffix.replace(".", "").upper()

    @computed_field
    def name(self) -> str:
        """Returns the name of the document."""
        return str(self.path.name)

    def read_text(self) -> str | None:
        """Reads and extracts text content from the document based on its type."""
        str_path = str(self.path)
        # Extract text based on document type
        match self.kind:
            case "PDF":
                return extract_text.from_pdf(str_path)
            case "DOCX":
                return extract_text.from_docx(str_path)
            case "HTML":
                return extract_text.from_html(str_path)
            case "HTM":
                return extract_text.from_html(str_path)
        return None


class DocumentList(BaseModel):
    """Model to hold a list of Document instances."""

    docs: list[Document]


def _collect_documents() -> DocumentList:
    """Collects documents from the 'data' directory and returns a DocumentList."""
    documents = []
    for pattern in ["**/*.pdf", "**/*.htm*", "**/*.docx"]:
        for path in pathlib.Path("data").glob(pattern):
            # Determine document metadata based on directory structure
            if len(path.parts) == 6:
                *_, journal, year, month_range, _ = path.parts
                documents.append(
                    Document(
                        journal=journal,
                        year=int(year),
                        month_range=month_range.upper(),
                        path=path,
                    )
                )
            else:
                _, _, journal, *_ = path.parts
                documents.append(
                    Document(journal=journal, year=None, month_range=None, path=path)
                )
    return DocumentList(docs=documents)


def _get_research_paper_content(content: str) -> ResearchPaperSummary | None:
    """Extracts research paper content including authors, title, and abstract."""
    extractor = ContentExtractor(
        content=content,
        response_model=ResearchPaperSummary,
        prompt_template="""
            Below is text extracted from a academic document:
            <document>
            {chunk}
            </document>
            Please extract the following information:
            {fields_to_extract}
            Provide the results in JSON format with keys: {json_keys}.
            """,
        model="gpt-4o-mini",
        base_url=None,
        chunk_step=300,
    )
    try:
        return extractor.extract_information()
    except ValidationError:
        # Return None if the validation of extracted content fails
        return None


def _get_date_range_metadata(content: str) -> ResearchPaperDates | None:
    """Extracts start and end dates from the document content."""
    extractor = ContentExtractor(
        content=content,
        prompt_template="""
        **Fields to extract from document:**
        {fields_to_extract}

        <document>
        {chunk}
        </document>

        **Instructions:**
        Extract the `start_date` and `end_date` from the provided document. Ensure that:
        - Dates are in the `YYYY-MM-DD` format.
        - `start_date` represents the first day of the starting month.
        - `end_date` represents the last day of the ending month.
        - If only a single month is provided, both `start_date` and `end_date` should correspond to that month.

        **Output Format:**
        Provide the results in JSON format ({json_keys}) adhering to the `PaperDates` model:
        ```json
        {{
            "start_date": "YYYY-MM-DD",
            "end_date": "YYYY-MM-DD"
        }}
        """,
        response_model=ResearchPaperDates,
        model="gpt-4o-mini",
        base_url=None,
        chunk_step=30,
    )
    try:
        return extractor.extract_information()
    except ValidationError:
        # Return None if the validation of extracted dates fails
        return None


if __name__ == "__main__":
    success_output = []
    failed_output = []
    docs = _collect_documents().docs

    # Load already processed documents from documents_success.jsonl
    already_processed_docs = set()
    try:
        with open("documents_success.jsonl") as success_file:
            for line in success_file:
                processed_doc = json.loads(line)
                already_processed_docs.add(processed_doc["path"])
    except FileNotFoundError:
        # If the file does not exist, skip loading and start fresh
        pass

    # Initialize locks for file writing
    success_file_lock = threading.Lock()
    failed_file_lock = threading.Lock()

    with Progress(
        SpinnerColumn(),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=CONSOLE,
    ) as progress:
        total_docs = len(docs)
        task = progress.add_task(
            f"Processed 0/{total_docs} documents", total=total_docs
        )

        def _process_document(
            doc: Document,
            progress,
            task_id,
            success_file_lock,
            failed_file_lock,
        ):
            """Processes an individual document by extracting metadata and content."""
            if str(doc.path) in already_processed_docs:
                # Skip processing if the document has already been processed
                LOGGER.info(
                    f"Document '{doc.path.name}' has already been processed. Skipping."
                )
                # Update progress
                progress.update(
                    task_id,
                    advance=1,
                    description=f"Processed {progress.tasks[task_id].completed}/{progress.tasks[task_id].total} documents",
                )
                return

            try:
                # Extract dates and content from the document
                dates = _get_date_range_metadata(
                    doc.model_dump_json(include={"year", "month_range"})
                )
                text_data = doc.read_text()
                if not text_data or len(text_data) < 100:
                    LOGGER.error(
                        f'Skipping since minimal text content extracted from "{doc.name}"'
                    )
                    raise FileNotFoundError()
                else:
                    summary = _get_research_paper_content(text_data)
                    if summary is None:
                        raise ValidationError()
                    processed_doc = (
                        doc.model_dump(mode="json")
                        | summary.model_dump()
                        | dates.model_dump(mode="json")
                    )
                    with (
                        success_file_lock,
                        open("documents_success.jsonl", "a") as success_file,
                    ):
                        success_file.write(json.dumps(processed_doc) + "\n")
            except ValidationError:
                # Handle validation error during extraction
                LOGGER.error(f"Validation error for document {doc}")
                failed_doc = doc.model_dump(mode="json")
                with (
                    failed_file_lock,
                    open("documents_failed.jsonl", "a") as failed_file,
                ):
                    failed_file.write(json.dumps(failed_doc) + "\n")
            except Exception as e:
                # Handle any unexpected errors during processing
                LOGGER.error(f"Unexpected error for document {doc}: {e}")
                failed_doc = doc.model_dump(mode="json")
                with (
                    failed_file_lock,
                    open("documents_failed.jsonl", "a") as failed_file,
                ):
                    failed_file.write(json.dumps(failed_doc) + "\n")
            finally:
                # Update progress
                progress.update(
                    task_id,
                    advance=1,
                    description=f"Processed {progress.tasks[task_id].completed}/{progress.tasks[task_id].total} documents",
                )

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    _process_document,
                    doc,
                    progress,
                    task,
                    success_file_lock,
                    failed_file_lock,
                )
                for doc in docs
            ]
            # Wait for all futures to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    LOGGER.error(f"Error processing document: {e}")
