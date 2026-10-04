from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.data.classes.results import RegressionHistory
from src.data.classes.settings import EnsembleMember, SimilarityRunConfig, SimilaritySettings
from src.data.loaders.similarity import PAD_TOKEN, SEP_TOKEN, build_loader, build_vocab, load_frame
from src.models.transformer import TransformerRegressor
from src.services.store import ResultStore
from src.utils.metrics import pearson_corr, rmse
from src.utils.plotting import apply_academic_style, build_styles, plot_learning_curves
from src.utils.progress import SIMILARITY, progress
from src.utils.seed import select_device, set_seed

logger = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "test")
HEAD_COUNTS = (1, 4, 8)
REGRESSION_PANELS = (
    ("val_loss", "Validation loss", "MSE"),
    ("val_pearson", "Validation Pearson", "Pearson correlation"),
)
PRESETS: dict[str, tuple[EnsembleMember, ...]] = {
    "single": (
        EnsembleMember(
            nhead=4,
            pooling="segment",
        ),
    ),
    "v9": (
        EnsembleMember(
            nhead=4,
            pooling="segment",
        ),
        EnsembleMember(
            nhead=8,
            pooling="segment",
        ),
        EnsembleMember(
            nhead=4,
            pooling="align_max",
        ),
        EnsembleMember(
            nhead=4,
            pooling="align",
        ),
        EnsembleMember(
            nhead=1,
            pooling="align_max",
            learning_rate=1e-3,
            weight_decay=0.05,
        ),
        EnsembleMember(
            nhead=8,
            pooling="align_max",
            learning_rate=1e-3,
            weight_decay=0.05,
        ),
        EnsembleMember(
            nhead=4,
            pooling="segment",
            head="softmax",
            learning_rate=1e-3,
            weight_decay=0.05,
        ),
        EnsembleMember(
            nhead=8,
            pooling="segment",
            head="softmax",
            learning_rate=1e-3,
            weight_decay=0.05,
        ),
        EnsembleMember(
            nhead=4,
            pooling="align_max",
            head="softmax",
            learning_rate=1e-3,
            weight_decay=0.05,
        ),
        EnsembleMember(
            nhead=4,
            pooling="align",
            head="softmax",
            learning_rate=1e-3,
            weight_decay=0.05,
        ),
        EnsembleMember(
            nhead=1,
            pooling="align_max",
            head="softmax",
            learning_rate=1e-3,
            weight_decay=0.05,
        ),
        EnsembleMember(
            nhead=8,
            pooling="align_max",
            head="softmax",
            learning_rate=1e-3,
            weight_decay=0.05,
        ),
    ),
}


def build_model(
    *,
    member: EnsembleMember,
    settings: SimilaritySettings,
    vocab: Mapping[str, int],
    device: torch.device,
) -> TransformerRegressor:
    return TransformerRegressor(
        vocab_size=len(vocab),
        embedding_dim=settings.embedding_dim,
        pad_idx=vocab[PAD_TOKEN],
        sep_idx=vocab[SEP_TOKEN],
        nhead=member.nhead,
        num_layers=settings.num_layers,
        dim_feedforward=settings.dim_feedforward,
        dropout=settings.dropout,
        max_len=settings.max_len,
        pooling=member.pooling,
        head=member.head,
    ).to(device=device)


def build_optimizer(
    *,
    model: nn.Module,
    settings: SimilaritySettings,
) -> optim.Optimizer:
    if settings.optimizer == "adam":
        return optim.Adam(
            params=model.parameters(),
            lr=settings.learning_rate,
        )
    return optim.AdamW(
        params=[
            {
                "params": [parameter for parameter in model.parameters() if parameter.dim() > 1],
                "weight_decay": settings.weight_decay,
            },
            {"params": [parameter for parameter in model.parameters() if parameter.dim() <= 1], "weight_decay": 0.0},
        ],
        lr=settings.learning_rate,
    )


def build_scheduler(
    *,
    optimizer: optim.Optimizer,
    settings: SimilaritySettings,
) -> optim.lr_scheduler.LRScheduler | optim.lr_scheduler.ReduceLROnPlateau | None:
    if settings.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=settings.epochs,
            eta_min=settings.min_learning_rate,
        )
    if settings.scheduler == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.5,
            patience=1,
        )
    return None


@torch.no_grad()
def predict(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs = [
        model(input_ids.to(device=device), attention_mask.to(device=device)).cpu().numpy()
        for input_ids, attention_mask, _ in loader
    ]
    return np.concatenate(outputs)


@torch.no_grad()
def evaluate_model(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    losses, scores, predictions = [], [], []
    for input_ids, attention_mask, batch_scores in loader:
        batch_scores = batch_scores.to(device=device)
        batch_predictions = model(input_ids.to(device=device), attention_mask.to(device=device))
        losses.append(criterion(batch_predictions, batch_scores).item())
        scores.extend(batch_scores.cpu().numpy())
        predictions.extend(batch_predictions.cpu().numpy())
    return (
        float(np.mean(losses)),
        pearson_corr(
            y_true=scores,
            y_pred=predictions,
        ),
        rmse(
            y_true=scores,
            y_pred=predictions,
        ),
    )


class BestPearsonCheckpoint:
    def __init__(self) -> None:
        self.best_score = -float("inf")
        self.state: dict[str, torch.Tensor] | None = None

    def update(
        self,
        *,
        score: float,
        model: nn.Module,
    ) -> None:
        if score > self.best_score:
            self.best_score = score
            self.state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    def restore(self, *, model: nn.Module) -> None:
        if self.state is not None:
            model.load_state_dict(state_dict=self.state)


def train_regressor(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    settings: SimilaritySettings,
    device: torch.device,
    description: str,
) -> RegressionHistory:
    criterion = nn.MSELoss()
    optimizer = build_optimizer(
        model=model,
        settings=settings,
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        settings=settings,
    )
    checkpoint = BestPearsonCheckpoint()
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_pearson": [], "val_rmse": []}

    with progress(
        stage=SIMILARITY,
        total=settings.epochs,
        detail=description,
        unit="epoch",
    ) as bar:
        for _ in range(settings.epochs):
            model.train()
            batch_losses = []
            for input_ids, attention_mask, scores in train_loader:
                optimizer.zero_grad()
                loss = criterion(
                    model(input_ids.to(device=device), attention_mask.to(device=device)), scores.to(device=device)
                )
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            val_loss, val_pearson, val_rmse = evaluate_model(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )
            for key, value in zip(history, (float(np.mean(batch_losses)), val_loss, val_pearson, val_rmse)):
                history[key].append(value)
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metrics=val_loss)
            elif scheduler is not None:
                scheduler.step()
            checkpoint.update(
                score=val_pearson,
                model=model,
            )
            bar.set_postfix_str(s=f"val Pearson {val_pearson:.4f}")
            bar.update(n=1)

    checkpoint.restore(model=model)
    return RegressionHistory(**{key: tuple(values) for key, values in history.items()})


class SimilarityPipeline:
    def __init__(
        self,
        *,
        config: SimilarityRunConfig,
        store: ResultStore,
    ) -> None:
        self.config = config
        self.store = store
        self.device = select_device(device_id=config.device_id)

    def prepare(self) -> tuple[dict[str, pd.DataFrame], dict[str, int], dict[str, DataLoader]]:
        frames = {
            name: load_frame(
                data_dir=self.config.data_dir,
                name=name,
            )
            for name in SPLIT_NAMES
        }
        vocab = build_vocab(frame=frames["train"])
        loaders = {
            name: build_loader(
                frame=frame,
                vocab=vocab,
                max_len=self.config.settings.max_len,
                batch_size=self.config.settings.batch_size,
                shuffle=name == "train",
            )
            for name, frame in frames.items()
        }
        logger.info("Vocabulary size %d, device %s", len(vocab), self.device)
        return frames, vocab, loaders

    def train_member(
        self,
        *,
        member: EnsembleMember,
        vocab: Mapping[str, int],
        loaders: Mapping[str, DataLoader],
    ) -> tuple[TransformerRegressor, RegressionHistory]:
        settings = member.resolve(settings=self.config.settings)
        set_seed(seed=self.config.seed)
        model = build_model(
            member=member,
            settings=settings,
            vocab=vocab,
            device=self.device,
        )
        history = train_regressor(
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            settings=settings,
            device=self.device,
            description=member.name,
        )
        return model, history

    def run(self) -> dict[str, float]:
        frames, vocab, loaders = self.prepare()
        val_true = frames["val"]["score"].to_numpy(dtype=np.float64)
        val_predictions, test_predictions, rows = {}, {}, []
        for member in self.config.members:
            model, history = self.train_member(
                member=member,
                vocab=vocab,
                loaders=loaders,
            )
            val_predictions[member.name] = predict(
                model=model,
                loader=loaders["val"],
                device=self.device,
            )
            test_predictions[member.name] = predict(
                model=model,
                loader=loaders["test"],
                device=self.device,
            )
            rows.append(
                {
                    "model": member.name,
                    "val_pearson": pearson_corr(
                        y_true=val_true,
                        y_pred=val_predictions[member.name],
                    ),
                    "val_rmse": rmse(
                        y_true=val_true,
                        y_pred=val_predictions[member.name],
                    ),
                    "best_epoch": history.best_epoch,
                }
            )
            logger.info("%s val Pearson %.4f", member.name, rows[-1]["val_pearson"])

        ensemble_val = np.mean(
            list(val_predictions.values()),
            axis=0,
        )
        summary = {
            "members": len(self.config.members),
            "ensemble_val_pearson": pearson_corr(
                y_true=val_true,
                y_pred=ensemble_val,
            ),
            "ensemble_val_rmse": rmse(
                y_true=val_true,
                y_pred=ensemble_val,
            ),
        }
        self.store.save_table(
            table=pd.DataFrame(data=rows).set_index(keys="model"),
            name="members",
        )
        submission = pd.DataFrame(
            data={
                "id": frames["test"]["id"],
                "score": np.mean(
                    list(test_predictions.values()),
                    axis=0,
                ),
            }
        )
        self.store.save_submission(submission=submission)
        self.store.save_summary(summary=summary)
        logger.info(
            "Ensemble val Pearson %.4f, submission -> %s", summary["ensemble_val_pearson"], self.store.submission_path
        )
        return summary

    def compare_heads(
        self,
        *,
        head_counts: Sequence[int] = HEAD_COUNTS,
        pooling: str = "segment",
    ) -> pd.DataFrame:
        frames, vocab, loaders = self.prepare()
        val_true = frames["val"]["score"].to_numpy(dtype=np.float64)
        rows, curves = [], {}
        for nhead in head_counts:
            member = EnsembleMember(
                nhead=nhead,
                pooling=pooling,
            )
            model, history = self.train_member(
                member=member,
                vocab=vocab,
                loaders=loaders,
            )
            predictions = predict(
                model=model,
                loader=loaders["val"],
                device=self.device,
            )
            rows.append(
                {
                    "nhead": nhead,
                    "d_head": self.config.settings.embedding_dim // nhead,
                    "val_pearson": pearson_corr(
                        y_true=val_true,
                        y_pred=predictions,
                    ),
                    "val_rmse": rmse(
                        y_true=val_true,
                        y_pred=predictions,
                    ),
                    "best_epoch": history.best_epoch,
                }
            )
            curves[f"h{nhead}"] = {"val_loss": history.val_loss, "val_pearson": history.val_pearson}

        table = pd.DataFrame(data=rows).set_index(keys="nhead")
        figure = plot_learning_curves(
            curves=curves,
            panels=REGRESSION_PANELS,
            styles=build_styles(names=list(curves)),
            labels={name: f"h = {name[1:]}" for name in curves},
        )
        self.store.save_figure(
            figure=figure,
            name="head_comparison",
        )
        self.store.save_table(
            table=table,
            name="head_comparison",
        )
        return table


def run_similarity(*, config: SimilarityRunConfig) -> dict[str, float]:
    apply_academic_style()
    store = ResultStore(root=config.output_dir)
    logger.info("Problem 2 outputs -> %s", store.root)
    return SimilarityPipeline(
        config=config,
        store=store,
    ).run()


def run_head_comparison(*, config: SimilarityRunConfig) -> pd.DataFrame:
    apply_academic_style()
    store = ResultStore(root=config.output_dir)
    logger.info("Problem 2 head comparison outputs -> %s", store.root)
    return SimilarityPipeline(
        config=config,
        store=store,
    ).compare_heads()
