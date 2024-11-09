from __future__ import annotations

import pathlib

from pydantic import BaseModel, computed_field

import extract_text


class ResearchPaperSummary(BaseModel):
    authors: str
    title: str
    abstract: str


class Document(BaseModel):
    journal: str
    year: int | None
    month_range: str | None
    path: pathlib.Path

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


if __name__ == "__main__":
    with open("documents.json", mode="w") as f:
        docs = collect_documents().model_dump_json(indent=4)
        f.write(docs)
