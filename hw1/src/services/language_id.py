from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.classes.language_id import LANGUAGE_CODES, LANGUAGES, Split, Vocabulary
from src.data.classes.results import Cost, Evaluation, TrainingHistory
from src.data.classes.settings import CellType, CharLMSettings, LanguageIdRunConfig
from src.data.loaders.language_id import build_split, build_vocabulary, load_split
from src.interface.language_model import LanguageModel
from src.models.char_lm import CharLanguageModel
from src.models.ngram import NGramLanguageModel
from src.services.store import ResultStore
from src.utils.measure import measure_cpu, measure_gpu
from src.utils.metrics import confusion_matrix, macro_f1
from src.utils.plotting import (
    apply_academic_style,
    build_styles,
    plot_accuracy_by_length,
    plot_confusion_matrices,
    plot_learning_curves,
    plot_perplexity_vs_accuracy,
    plot_radar,
)
from src.utils.progress import LSTM, RNN, Stage, progress
from src.utils.seed import select_device, set_seed

logger = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "test")
NGRAM_ORDERS = {"Bigram": 2, "Trigram": 3}
CELL_NAMES: dict[CellType, str] = {"rnn": "RNN", "lstm": "LSTM"}
CELL_STAGES: dict[CellType, Stage] = {"rnn": RNN, "lstm": LSTM}
MODEL_NAMES = ("Bigram", "Trigram", "RNN", "LSTM")
LENGTH_BINS = (0, 30, 60, 120, np.inf)
LENGTH_LABELS = ("<=30", "31-60", "61-120", ">120")
LEARNING_PANELS = (
    ("train_loss", "Training loss", "Cross-entropy (nats/char)"),
    ("val_ppl", "Validation perplexity", "Perplexity"),
)


def score(
    *,
    models: Mapping[str, LanguageModel],
    split: Split,
) -> tuple[np.ndarray, np.ndarray]:
    nll = np.stack(
        arrays=[models[language].sequence_nll(sequences=split.sequences) for language in LANGUAGES],
        axis=1,
    )
    return nll, np.array(
        [len(ids) + 1 for ids in split.sequences],
        dtype=float,
    )


def build_prediction_table(
    *,
    nll: np.ndarray,
    lengths: np.ndarray,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    perplexity = np.exp(nll / lengths[:, None])
    table = frame[["Text", "Language"]].assign(
        Predicted=[LANGUAGES[index] for index in perplexity.argmin(axis=1)],
        Length=frame["Text"].str.len(),
        **{f"PPL_{language}": perplexity[:, index] for index, language in enumerate(LANGUAGES)},
    )
    table["Correct"] = table["Language"] == table["Predicted"]
    return table


def evaluate(
    *,
    nll: np.ndarray,
    lengths: np.ndarray,
    frame: pd.DataFrame,
) -> Evaluation:
    table = build_prediction_table(
        nll=nll,
        lengths=lengths,
        frame=frame,
    )
    true_columns = np.array([LANGUAGES.index(language) for language in frame["Language"]])
    matrix = confusion_matrix(
        y_true=table["Language"],
        y_pred=table["Predicted"],
        labels=LANGUAGES,
    )
    return Evaluation(
        accuracy=float(table["Correct"].mean()),
        macro_f1=macro_f1(matrix=matrix),
        corpus_ppl=math.exp(nll[np.arange(len(frame)), true_columns].sum() / lengths.sum()),
        confusion=matrix,
        table=table,
    )


def evaluate_split(
    *,
    models: Mapping[str, LanguageModel],
    split: Split,
) -> Evaluation:
    nll, lengths = score(
        models=models,
        split=split,
    )
    return evaluate(
        nll=nll,
        lengths=lengths,
        frame=split.frame,
    )


def summarize(*, evaluations: Mapping[str, Evaluation]) -> pd.DataFrame:
    rows = {
        name: {"accuracy": evaluation.accuracy, "macro_f1": evaluation.macro_f1, "corpus_ppl": evaluation.corpus_ppl}
        for name, evaluation in evaluations.items()
    }
    return pd.DataFrame(data=rows).T


def accuracy_by_length(*, table: pd.DataFrame) -> pd.DataFrame:
    buckets = pd.cut(
        x=table["Length"],
        bins=LENGTH_BINS,
        labels=LENGTH_LABELS,
    )
    return table.groupby(
        by=buckets,
        observed=False,
    )["Correct"].agg(
        accuracy="mean",
        count="count",
    )


def confused_pairs(*, matrix: np.ndarray) -> pd.DataFrame:
    errors = matrix - np.diag(np.diag(matrix))
    pairs = {
        f"{LANGUAGES[i]} <-> {LANGUAGES[j]}": errors[i, j] + errors[j, i]
        for i in range(len(LANGUAGES))
        for j in range(i + 1, len(LANGUAGES))
    }
    return (
        pd.Series(
            data=pairs,
            name="errors",
        )
        .sort_values(ascending=False)
        .to_frame()
    )


def decision_margins(*, table: pd.DataFrame) -> np.ndarray:
    log_ppl = np.log(table[[f"PPL_{language}" for language in LANGUAGES]].to_numpy())
    is_true = np.eye(
        len(LANGUAGES),
        dtype=bool,
    )[[LANGUAGES.index(language) for language in table["Language"]]]
    return np.where(is_true, np.inf, log_ppl).min(axis=1) - log_ppl[is_true]


def per_language_accuracy(*, evaluation: Evaluation) -> np.ndarray:
    return np.diag(evaluation.confusion) / np.maximum(evaluation.confusion.sum(axis=1), 1)


def train_ngram_models(
    *,
    n: int,
    split: Split,
    vocabulary: Vocabulary,
) -> dict[str, NGramLanguageModel]:
    return {
        language: NGramLanguageModel(
            n=n,
            vocabulary=vocabulary,
        ).fit(sequences=split.by_language[language])
        for language in LANGUAGES
    }


class BestCheckpoint:
    def __init__(self) -> None:
        self.best_score = float("inf")
        self.state: dict[str, torch.Tensor] | None = None

    def update(
        self,
        *,
        score: float,
        model: nn.Module,
    ) -> None:
        if score < self.best_score:
            self.best_score = score
            self.state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    def restore(self, *, model: nn.Module) -> None:
        if self.state is None:
            raise RuntimeError("no checkpoint was recorded")
        model.load_state_dict(state_dict=self.state)


def corpus_ppl(
    *,
    model: CharLanguageModel,
    sequences: Sequence[list[int]],
) -> float:
    return math.exp(model.sequence_nll(sequences=sequences).sum() / sum(len(ids) + 1 for ids in sequences))


def run_epoch(
    *,
    model: CharLanguageModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
) -> float:
    model.train()
    total_loss, total_tokens = 0.0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device=model.device), targets.to(device=model.device)
        n_tokens = int((targets != model.vocabulary.pad_id).sum())
        loss = (
            model.token_nll(
                inputs=inputs,
                targets=targets,
            ).sum()
            / n_tokens
        )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            parameters=model.parameters(),
            max_norm=grad_clip,
        )
        optimizer.step()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
    return total_loss / total_tokens


def train_language_model(
    *,
    cell: CellType,
    language: str,
    train_sequences: Sequence[list[int]],
    val_sequences: Sequence[list[int]],
    settings: CharLMSettings,
    vocabulary: Vocabulary,
    device: torch.device,
    seed: int,
) -> tuple[CharLanguageModel, TrainingHistory]:
    set_seed(seed=seed)
    model = CharLanguageModel(
        cell=cell,
        settings=settings,
        vocabulary=vocabulary,
    ).to(device=device)
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=settings.learning_rate,
    )
    loader = DataLoader(
        dataset=list(train_sequences),
        batch_size=settings.batch_size,
        shuffle=True,
        collate_fn=model.collate,
        generator=torch.Generator().manual_seed(seed),
    )
    checkpoint = BestCheckpoint()
    train_loss, val_ppl = [], []

    with progress(
        stage=CELL_STAGES[cell],
        total=settings.epochs,
        detail=language,
        unit="epoch",
    ) as bar:
        for _ in range(settings.epochs):
            train_loss.append(
                run_epoch(
                    model=model,
                    loader=loader,
                    optimizer=optimizer,
                    grad_clip=settings.grad_clip,
                )
            )
            val_ppl.append(
                corpus_ppl(
                    model=model,
                    sequences=val_sequences,
                )
            )
            checkpoint.update(
                score=val_ppl[-1],
                model=model,
            )
            bar.set_postfix_str(s=f"loss {train_loss[-1]:.3f}, val PPL {val_ppl[-1]:.3f}")
            bar.update(n=1)

    checkpoint.restore(model=model)
    return model, TrainingHistory(
        train_loss=tuple(train_loss),
        val_ppl=tuple(val_ppl),
    )


def train_all_languages(
    *,
    cell: CellType,
    settings: CharLMSettings,
    train_data: Split,
    val_data: Split,
    vocabulary: Vocabulary,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, CharLanguageModel], dict[str, TrainingHistory], Cost]:
    def train_every_language() -> dict[str, tuple[CharLanguageModel, TrainingHistory]]:
        return {
            language: train_language_model(
                cell=cell,
                language=language,
                train_sequences=train_data.by_language[language],
                val_sequences=val_data.by_language[language],
                settings=settings,
                vocabulary=vocabulary,
                device=device,
                seed=seed,
            )
            for language in LANGUAGES
        }

    trained, cost = measure_gpu(
        task=train_every_language,
        device=device,
    )
    models = {language: model for language, (model, _) in trained.items()}
    histories = {language: history for language, (_, history) in trained.items()}
    return models, histories, cost


def evaluate_neural(
    *,
    models: Mapping[str, CharLanguageModel],
    val_data: Split,
    test_data: Split,
    device: torch.device,
) -> tuple[Evaluation, Evaluation, Cost]:
    val_evaluation = evaluate_split(
        models=models,
        split=val_data,
    )
    (nll, lengths), cost = measure_gpu(
        task=partial(
            score,
            models=models,
            split=test_data,
        ),
        device=device,
    )
    return (
        val_evaluation,
        evaluate(
            nll=nll,
            lengths=lengths,
            frame=test_data.frame,
        ),
        cost,
    )


def count_parameters(*, models: Mapping[str, nn.Module]) -> int:
    return sum(parameter.numel() for model in models.values() for parameter in model.parameters())


class LanguageIdPipeline:
    def __init__(
        self,
        *,
        config: LanguageIdRunConfig,
        store: ResultStore,
    ) -> None:
        self.config = config
        self.store = store
        self.device = select_device(device_id=config.device_id)
        self.language_styles = build_styles(names=LANGUAGES)
        self.model_styles = build_styles(names=MODEL_NAMES)

    def run(self) -> dict[str, dict[str, float]]:
        set_seed(seed=self.config.seed)
        frames = {
            name: load_split(
                data_dir=self.config.data_dir,
                name=name,
            )
            for name in SPLIT_NAMES
        }
        vocabulary = build_vocabulary(texts=frames["train"]["Text"].tolist())
        splits = {
            name: build_split(
                frame=frame,
                vocabulary=vocabulary,
            )
            for name, frame in frames.items()
        }
        logger.info("Vocabulary size %d, device %s", vocabulary.size, self.device)

        val_results, test_results, costs, sizes = self.run_ngrams(
            splits=splits,
            vocabulary=vocabulary,
        )
        for cell in self.config.cells:
            name = CELL_NAMES[cell]
            models, histories, train_cost = train_all_languages(
                cell=cell,
                settings=self.config.char_lm,
                train_data=splits["train"],
                val_data=splits["val"],
                vocabulary=vocabulary,
                device=self.device,
                seed=self.config.seed,
            )
            val_results[name], test_results[name], infer_cost = evaluate_neural(
                models=models,
                val_data=splits["val"],
                test_data=splits["test"],
                device=self.device,
            )
            costs[name] = (train_cost, infer_cost, "GPU")
            sizes[name] = count_parameters(models=models)
            self.save_learning_curves(
                histories=histories,
                name=name,
            )
            logger.info(
                "%s test accuracy %.4f, perplexity %.3f",
                name,
                test_results[name].accuracy,
                test_results[name].corpus_ppl,
            )

        return self.save_comparison(
            val_results=val_results,
            test_results=test_results,
            costs=costs,
            sizes=sizes,
        )

    def run_ngrams(
        self,
        *,
        splits: Mapping[str, Split],
        vocabulary: Vocabulary,
    ) -> tuple[dict[str, Evaluation], dict[str, Evaluation], dict[str, tuple[Cost, Cost, str]], dict[str, int]]:
        val_results, test_results, costs, sizes = {}, {}, {}, {}
        for name, n in NGRAM_ORDERS.items():
            models, build_cost = measure_cpu(
                task=partial(
                    train_ngram_models,
                    n=n,
                    split=splits["train"],
                    vocabulary=vocabulary,
                ),
            )
            val_results[name] = evaluate_split(
                models=models,
                split=splits["val"],
            )
            (nll, lengths), infer_cost = measure_cpu(
                task=partial(
                    score,
                    models=models,
                    split=splits["test"],
                )
            )
            test_results[name] = evaluate(
                nll=nll,
                lengths=lengths,
                frame=splits["test"].frame,
            )
            costs[name] = (build_cost, infer_cost, "CPU")
            sizes[name] = sum(model.stored_counts for model in models.values())
            logger.info(
                "%s test accuracy %.4f, perplexity %.3f",
                name,
                test_results[name].accuracy,
                test_results[name].corpus_ppl,
            )
        return val_results, test_results, costs, sizes

    def save_learning_curves(
        self,
        *,
        histories: Mapping[str, TrainingHistory],
        name: str,
    ) -> None:
        curves = {
            language: {"train_loss": history.train_loss, "val_ppl": history.val_ppl}
            for language, history in histories.items()
        }
        figure = plot_learning_curves(
            curves=curves,
            panels=[(field, f"{name} {title.lower()}", ylabel) for field, title, ylabel in LEARNING_PANELS],
            styles=self.language_styles,
            labels=LANGUAGE_CODES,
        )
        self.store.save_figure(
            figure=figure,
            name=f"{name.lower()}_learning_curves",
        )
        selection = pd.DataFrame(
            data={
                language: {"best_epoch": history.best_epoch, "best_val_ppl": history.best_val_ppl}
                for language, history in histories.items()
            }
        ).T
        self.store.save_table(
            table=selection,
            name=f"{name.lower()}_checkpoint_selection",
        )

    def save_comparison(
        self,
        *,
        val_results: Mapping[str, Evaluation],
        test_results: Mapping[str, Evaluation],
        costs: Mapping[str, tuple[Cost, Cost, str]],
        sizes: Mapping[str, int],
    ) -> dict[str, dict[str, float]]:
        comparison = pd.DataFrame(
            data={
                name: {
                    "hardware": hardware,
                    "stored counts / parameters": sizes[name],
                    "train/build time (s)": train_cost.seconds,
                    "train/build peak memory (MB)": train_cost.peak_mb,
                    "inference time (s)": infer_cost.seconds,
                    "inference peak memory (MB)": infer_cost.peak_mb,
                    "val corpus PPL": val_results[name].corpus_ppl,
                    "test corpus PPL": test_results[name].corpus_ppl,
                    "test accuracy": test_results[name].accuracy,
                    "test macro-F1": test_results[name].macro_f1,
                }
                for name, (train_cost, infer_cost, hardware) in costs.items()
            }
        ).T
        self.store.save_table(
            table=comparison,
            name="comparison",
        )

        margins = {name: decision_margins(table=evaluation.table) for name, evaluation in test_results.items()}
        pair_table = pd.concat(
            objs={
                name: confused_pairs(matrix=evaluation.confusion)["errors"] for name, evaluation in test_results.items()
            },
            axis=1,
        )
        length_table = pd.concat(
            objs={
                name: accuracy_by_length(table=evaluation.table)["accuracy"]
                for name, evaluation in test_results.items()
            },
            axis=1,
        )
        self.store.save_table(
            table=pair_table,
            name="confused_pairs",
        )
        self.store.save_table(
            table=length_table,
            name="accuracy_by_length",
        )

        tick_labels = [LANGUAGE_CODES[language] for language in LANGUAGES]
        figures = {
            "comparison_confusion_matrices": plot_confusion_matrices(
                matrices={name: evaluation.confusion for name, evaluation in test_results.items()},
                tick_labels=tick_labels,
            ),
            "comparison_perplexity_accuracy": plot_perplexity_vs_accuracy(
                points={
                    name: (evaluation.corpus_ppl, evaluation.accuracy) for name, evaluation in test_results.items()
                },
                margins=margins,
                styles=self.model_styles,
            ),
            "comparison_accuracy_by_length": plot_accuracy_by_length(
                length_table=length_table,
                styles=self.model_styles,
            ),
            "comparison_language_radar": plot_radar(
                values={
                    name: per_language_accuracy(evaluation=evaluation) for name, evaluation in test_results.items()
                },
                axis_labels=tick_labels,
                legend_labels={
                    name: f"{name} (overall {evaluation.accuracy:.3f})" for name, evaluation in test_results.items()
                },
                styles=self.model_styles,
            ),
        }
        for name, figure in figures.items():
            self.store.save_figure(
                figure=figure,
                name=name,
            )

        summary = {
            name: summarize(evaluations={name: evaluation}).iloc[0].to_dict()
            for name, evaluation in test_results.items()
        }
        self.store.save_summary(summary=summary)
        return summary


def run_language_id(*, config: LanguageIdRunConfig) -> dict[str, dict[str, float]]:
    apply_academic_style()
    store = ResultStore(root=config.output_dir)
    logger.info("Problem 1 outputs -> %s", store.root)
    return LanguageIdPipeline(
        config=config,
        store=store,
    ).run()
