from __future__ import annotations

import pathlib
import time

import instructor
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from rich import print_json
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from extract_text import from_pdf
from parse_documents import ResearchPaperSummary
from print_utils import CONSOLE, LOGGER


class ContentExtractor:
    def __init__(
        self,
        content: str,
        prompt_template: str,
        response_model: BaseModel,
        client: instructor.Client = None,
        model: str = "llama3.2",
        chunk_step: int = 250,
        max_chunks: int = 20,
        max_retries: int = 10,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "default_api_key",
    ):
        self.content = content
        self.client = client or self.create_default_client(base_url, api_key)
        self.response_model = response_model
        self.model = model
        self.extraction_prompt_template = prompt_template
        self.chunk_step = chunk_step
        self.max_chunks = max_chunks
        self.max_retries = max_retries
        self.console = CONSOLE  # Use the shared console
        self.words = self.content.split()
        self.total_words = len(self.words)

        # Initialize a dictionary to store extracted fields
        self.extracted_data = {"authors": None, "title": None, "abstract": None}

    @staticmethod
    def create_default_client(base_url: str, api_key: str) -> instructor.Client:
        return instructor.from_openai(
            OpenAI(
                base_url=base_url,
                api_key=api_key,
            ),
            mode=instructor.Mode.JSON,
        )

    def _get_cumulative_chunk(self, current_step: int) -> str:
        """Returns the cumulative chunk up to the current step."""
        end = min(self.chunk_step * current_step, self.total_words)
        return " ".join(self.words[:end])

    def extract_information(self) -> BaseModel | None:
        # Initialize Rich Progress with the shared console
        with Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[bold green]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,  # Remove the progress bar after completion
        ) as progress:
            progress_task = progress.add_task(
                f"Processing chunks of '[blue]{pathlib.Path(pdf_path).name[20:]}[/blue]'...",
                total=self.max_chunks,
            )
            overall_start_time = time.time()

            # List of fields to extract
            fields = ["authors", "title", "abstract"]

            for step in range(1, self.max_chunks + 1):
                current_chunk_size = self.chunk_step * step
                chunk = self._get_cumulative_chunk(step)

                # Determine which fields still need to be extracted
                missing_fields = [
                    field for field in fields if self.extracted_data[field] is None
                ]

                # If all fields have been extracted, break the loop
                if not missing_fields:
                    break

                # Create dynamic prompt based on missing fields
                fields_to_extract_str = "\n".join(
                    f"- {field.capitalize()}" for field in missing_fields
                )
                json_keys_str = ", ".join(missing_fields)

                extraction_prompt = self.extraction_prompt_template.format(
                    chunk=chunk,
                    fields_to_extract=fields_to_extract_str,
                    json_keys=json_keys_str,
                )

                chunk_start_time = time.time()

                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "user",
                                "content": extraction_prompt,
                            }
                        ],
                        response_model=self.response_model,
                        max_retries=self.max_retries,
                    )
                    LOGGER.debug(f"Response: {resp}")
                except ValidationError as e:
                    LOGGER.warning(f"Validation error: {e}")
                    progress.advance(progress_task)
                    continue

                chunk_end_time = time.time()
                elapsed_time = chunk_end_time - chunk_start_time
                LOGGER.info(f"Chunk {step} processed in {elapsed_time:.2f} seconds.")

                # Update extracted data with any new fields
                for field in missing_fields:
                    value = getattr(resp, field, None)
                    if value:
                        self.extracted_data[field] = value
                        LOGGER.info(
                            f"Extracted '{field}' at chunk size {current_chunk_size} words."
                        )

                # Advance the progress bar after processing each chunk
                progress.advance(progress_task)

            overall_end_time = time.time()
            total_elapsed = overall_end_time - overall_start_time

            # Check if all fields have been extracted
            if all(self.extracted_data.values()):
                LOGGER.info(
                    f"Successfully extracted all required information in {total_elapsed:.2f} seconds."
                )
                # Create an instance of response_model with the extracted data
                return self.response_model(**self.extracted_data)
            else:
                missing_fields = [
                    field for field in fields if self.extracted_data[field] is None
                ]
                LOGGER.error(
                    f"Failed to extract all required information within the maximum chunk size after {total_elapsed:.2f} seconds."
                )
                LOGGER.error(f"Missing fields: {missing_fields}")
                # You can return the partially filled model or None
                return self.response_model(**self.extracted_data)


if __name__ == "__main__":
    pdf_path = "data/journals/Acta Veterinaria Scandinavica/2017/Jan-Feb/Abscess of the cervical spine secondary to injection site infection in a heifer.pdf"
    content = from_pdf(pdf_path)
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
    )
    extracted_content = extractor.extract_information()
    print_json(extracted_content.model_dump_json())