from __future__ import annotations

import json
import pathlib
from datetime import date

from pydantic import BaseModel, Field, computed_field
from pydantic_core import ValidationError
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

import extract_text
from print_utils import CONSOLE, LOGGER
from process_llm import ContentExtractor


class ResearchPaperSummary(BaseModel):
    authors: list[str]
    title: str
    abstract: str


class ResearchPaperDates(BaseModel):
    start_date: date
    end_date: date


class Document(BaseModel):
    journal: str
    year: int | None
    month_range: str | None
    path: pathlib.Path = Field(
        ...,
    )

    @computed_field
    def kind(self) -> str:
        return self.path.suffix.replace(".", "").upper()

    @computed_field
    def name(self) -> str:
        return str(self.path.name)

    def read_text(self) -> str | None:
        str_path = str(self.path)
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
    docs: list[Document]


def collect_documents() -> DocumentList:
    documents = []
    for pattern in ["**/*.pdf", "**/*.htm*", "**/*.docx"]:
        for path in pathlib.Path("data").glob(pattern):
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
                _, journal, *_ = path.parts
                documents.append(
                    Document(journal=journal, year=None, month_range=None, path=path)
                )
    return DocumentList(docs=documents)


def get_research_paper_content(content: str) -> ResearchPaperSummary | None:
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
        model="gpt-4o",
        base_url=None,
        chunk_step=300,
    )
    try:
        return extractor.extract_information()
    except ValidationError:
        return None


def get_date_range_metadata(content: str) -> ResearchPaperDates | None:
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
        model="gpt-4o",
        base_url=None,
        chunk_step=30,
    )
    try:
        return extractor.extract_information()
    except ValidationError:
        return None



if __name__ == "__main__":
    success_output = []
    failed_output = []
    docs = collect_documents().docs

    with Progress(
        SpinnerColumn(),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=CONSOLE,
    ) as progress:
        total_docs = len(docs)
        task = progress.add_task(
            f"Processed 0/{total_docs} documents", total=total_docs
        )

        for index, doc in enumerate(docs):
            progress.update(
                task, description=f"Processing {index+1}/{total_docs} documents"
            )
            try:
                dates = get_date_range_metadata(
                    doc.model_dump_json(include={"year", "month_range"})
                )
                summary = get_research_paper_content(doc.read_text())
                processed_doc = (
                    doc.model_dump(mode="json")
                    | summary.model_dump()
                    | dates.model_dump(mode="json")
                )
                success_output.append(processed_doc)
                # Append to documents_success.jsonl
                with open("documents_success.jsonl", "a") as success_file:
                    success_file.write(json.dumps(processed_doc) + "\n")
            except ValidationError:
                failed_output.append(doc.model_dump(mode="json"))
                LOGGER.error(f"Validation error for document {doc}")
                # Append to documents_failed.jsonl
                with open("documents_failed.jsonl", "a") as failed_file:
                    failed_file.write(json.dumps(doc.model_dump(mode="json")) + "\n")
            except Exception as e:
                failed_output.append(doc.model_dump(mode="json"))
                LOGGER.error(f"Unexpected error for document {doc}: {e}")
                # Append to documents_failed.jsonl
                with open("documents_failed.jsonl", "a") as failed_file:
                    failed_file.write(json.dumps(doc.model_dump(mode="json")) + "\n")
            finally:
                progress.advance(task)
