from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.plotting import save_figure

FIGURES_DIR_NAME = "figures"
TABLES_DIR_NAME = "tables"
SUMMARY_FILE_NAME = "summary.json"
SUBMISSION_FILE_NAME = "submission.csv"


class ResultStore:
    def __init__(self, *, root: Path) -> None:
        self.root = root
        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

    @property
    def submission_path(self) -> Path:
        return self.root / SUBMISSION_FILE_NAME

    def save_figure(
        self,
        *,
        figure: plt.Figure,
        name: str,
    ) -> Path:
        path = self.root / FIGURES_DIR_NAME / name
        save_figure(
            figure=figure,
            path=path,
        )
        plt.close(fig=figure)
        return path

    def save_table(
        self,
        *,
        table: pd.DataFrame,
        name: str,
    ) -> Path:
        path = self.root / TABLES_DIR_NAME / f"{name}.csv"
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        table.to_csv(path_or_buf=path)
        return path

    def save_summary(self, *, summary: Mapping[str, object]) -> Path:
        path = self.root / SUMMARY_FILE_NAME
        path.write_text(
            data=json.dumps(
                obj=summary,
                indent=2,
                default=float,
            ),
            encoding="utf-8",
        )
        return path

    def save_submission(self, *, submission: pd.DataFrame) -> Path:
        submission.to_csv(
            path_or_buf=self.submission_path,
            index=False,
        )
        return self.submission_path
