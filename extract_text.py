from __future__ import annotations

import pathlib

import pdfplumber
from bs4 import BeautifulSoup
from docx import Document

from print_utils import LOGGER


def from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    LOGGER.info(
        f'Extracted {len(text)} chars from "{pathlib.Path(pdf_path).name[20:]}..."'
    )
    return text


def from_docx(docx_path: str) -> str:
    doc = Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    LOGGER.info(
        f'Extracted {len(text)} chars from "{pathlib.Path(docx_path).name[20:]}..."'
    )
    return text


def from_html(html_path: str) -> str:
    with open(html_path, encoding="utf-8") as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator="\n")
    LOGGER.info(
        f'Extracted {len(text)} chars from "{pathlib.Path(html_content).name[20:]}..."'
    )
    return text


if __name__ == "__main__":
    print(
        from_pdf(
            "data/journals/Acta Veterinaria Scandinavica/2017/Jan-Feb/Abscess of the cervical spine secondary to injection site infection in a heifer.pdf"
        )
    )
