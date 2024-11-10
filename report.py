from __future__ import annotations

import pandas as pd

if __name__ == "__main__":
    success_df = pd.read_json("documents_success.jsonl", lines=True).assign(
        start_date=lambda df: pd.to_datetime(df["start_date"], errors="coerce"),
        end_date=lambda df: pd.to_datetime(df["end_date"], errors="coerce"),
    )
    failure_df = pd.read_json("documents_failed.jsonl", lines=True)

    with pd.ExcelWriter("documents_combined.xlsx", engine="openpyxl") as writer:
        success_df.to_excel(writer, sheet_name="Success", index=False)
        failure_df.to_excel(writer, sheet_name="Failure", index=False)
