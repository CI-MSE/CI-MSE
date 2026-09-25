"""Training-time Critical Interval MSE evaluation for LeRobotDataset."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .critical_interval_mse import CriticalIntervalMSE, CriticalIntervalMSEConfig


def _as_int(value: Any) -> int:
    return int(value.item()) if hasattr(value, "item") else int(value)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _simple_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {}
    out: dict[str, Any] = {}
    for key in samples[0]:
        vals = [sample[key] for sample in samples]
        try:
            out[key] = np.stack(vals)
        except Exception:
            out[key] = vals
    return out


def _default_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from torch.utils.data._utils.collate import default_collate

        return default_collate(samples)
    except Exception:
        return _simple_collate(samples)


def _move_to_device(batch: Any, device: str | None) -> Any:
    if device is None:
        return batch
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: _move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [_move_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(_move_to_device(value, device) for value in batch)
    return batch


def _no_grad_context(enabled: bool):
    if not enabled:
        return nullcontext()
    try:
        import torch

        return torch.no_grad()
    except Exception:
        return nullcontext()


def _summary(errors: np.ndarray) -> dict[str, float | int]:
    if len(errors) == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "q1": float("nan"),
            "q99": float("nan"),
            "mean_1_99": float("nan"),
            "num_timesteps": 0,
        }
    q1 = float(np.percentile(errors, 1))
    q99 = float(np.percentile(errors, 99))
    mask = (errors >= q1) & (errors <= q99)
    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "q1": q1,
        "q99": q99,
        "mean_1_99": float(np.mean(errors[mask])) if mask.any() else float(np.mean(errors)),
        "num_timesteps": int(len(errors)),
    }


@dataclass
class LeRobotCriticalIntervalMSEConfig:
    """Configuration for training-time LeRobot CI-MSE evaluation."""

    ci_mse_config: CriticalIntervalMSEConfig = field(default_factory=CriticalIntervalMSEConfig)
    batch_size: int = 64
    target_key: str = "action"
    device: str | None = None
    action_horizon: int | None = None
    no_grad: bool = True
    episodes: list[int] | None = None


class LeRobotCriticalIntervalMSE:
    """Evaluate CI-MSE during training from a LeRobotDataset and predict_fn.

    This class composes the array-based :class:`CriticalIntervalMSE` evaluator
    with LeRobotDataset index selection and batching.
    """

    def __init__(
        self,
        dataset: Any,
        intervals_path: str | Path,
        config: LeRobotCriticalIntervalMSEConfig | None = None,
        collate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
    ) -> None:
        self.dataset = dataset
        self.intervals_path = Path(intervals_path)
        self.config = config or LeRobotCriticalIntervalMSEConfig()
        self.collate_fn = collate_fn or _default_collate
        self.evaluator = CriticalIntervalMSE(self.config.ci_mse_config)
        self.critical_intervals = self._load_critical_intervals()

    def required_indices(self) -> list[int]:
        """Return sorted global LeRobot indices required for this validation run."""
        action_horizon = self._infer_action_horizon()
        if action_horizon is None:
            return []

        lookback = self._lookback(action_horizon)
        required: set[int] = set()
        for ep_idx, intervals in self._iter_episode_intervals():
            ep_start, ep_end = self._episode_bounds(ep_idx)
            ep_len = ep_end - ep_start
            for start, end in intervals:
                s = max(0, min(ep_len, int(start)))
                t = max(0, min(ep_len, int(end)))
                if t <= s:
                    continue
                for local_idx in range(max(0, s - lookback), t):
                    required.add(ep_start + local_idx)
        return sorted(required)

    def evaluate(self, predict_fn: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
        """Run predict_fn over required validation samples and return CI-MSE metrics."""
        required = self.required_indices()
        if not required:
            return self._empty_result()

        predictions_by_index, targets_by_index = self._predict_indices(required, predict_fn)
        return self._compute_from_index_maps(predictions_by_index, targets_by_index)

    def evaluate_distributed(
        self,
        predict_fn: Callable[[dict[str, Any]], Any],
        process_group: Any = None,
        broadcast_result: bool = True,
    ) -> dict[str, Any] | None:
        """Evaluate CI-MSE in a torch.distributed job using all_gather_object.

        If torch.distributed is unavailable or not initialized, this falls back
        to :meth:`evaluate`. In distributed mode, every rank predicts a shard of
        required indices; rank 0 gathers all predictions and computes CI-MSE on
        the restored episode timeline. When ``broadcast_result`` is true, scalar
        summary fields are broadcast to all ranks. Raw arrays remain available
        only on rank 0.
        """
        try:
            import torch.distributed as dist
        except Exception:
            return self.evaluate(predict_fn)

        if not dist.is_available() or not dist.is_initialized():
            return self.evaluate(predict_fn)

        rank = dist.get_rank(process_group)
        world_size = dist.get_world_size(process_group)
        required = self.required_indices()
        local_required = required[rank::world_size]
        predictions_by_index, targets_by_index = self._predict_indices(
            local_required,
            predict_fn,
        )

        payload = {
            "indices": local_required,
            "predictions": [predictions_by_index[idx] for idx in local_required],
            "targets": [targets_by_index[idx] for idx in local_required],
        }
        gathered: list[Any] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, payload, group=process_group)

        result = None
        if rank == 0:
            gathered_predictions: dict[int, np.ndarray] = {}
            gathered_targets: dict[int, np.ndarray] = {}
            for item in gathered:
                for idx, pred, target in zip(
                    item["indices"],
                    item["predictions"],
                    item["targets"],
                ):
                    if idx in gathered_predictions:
                        raise ValueError(f"Duplicate gathered validation index: {idx}")
                    gathered_predictions[int(idx)] = np.asarray(pred)
                    gathered_targets[int(idx)] = np.asarray(target)
            missing = sorted(set(required) - set(gathered_predictions))
            if missing:
                raise ValueError(f"Missing gathered validation indices: {missing[:10]}")
            result = self._compute_from_index_maps(gathered_predictions, gathered_targets)

        if not broadcast_result:
            return result

        scalar_result = None
        if rank == 0:
            scalar_result = {
                key: result[key]
                for key in ("mean", "median", "q1", "q99", "mean_1_99", "num_timesteps")
            }
        holder = [scalar_result]
        dist.broadcast_object_list(holder, src=0, group=process_group)
        if rank == 0:
            return result
        return holder[0]

    def _empty_result(self) -> dict[str, Any]:
        empty = np.array([], dtype=np.float64)
        return {
            **_summary(empty),
            "errors": empty,
            "episode_ends": np.array([], dtype=np.int64),
            "interval_ends": np.empty((0, 0), dtype=np.int64),
            "episode_indices": [],
            "episode_spans": [],
        }

    def _predict_indices(
        self,
        indices: list[int],
        predict_fn: Callable[[dict[str, Any]], Any],
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        predictions_by_index: dict[int, np.ndarray] = {}
        targets_by_index: dict[int, np.ndarray] = {}
        for batch_indices in self._chunks(indices, self.config.batch_size):
            samples = [self.dataset[idx] for idx in batch_indices]
            batch = self.collate_fn(samples)
            batch = _move_to_device(batch, self.config.device)

            with _no_grad_context(self.config.no_grad):
                pred = predict_fn(batch)

            pred_np = _to_numpy(pred)
            target_np = _to_numpy(batch[self.config.target_key])
            if pred_np.ndim != 3:
                raise ValueError(
                    f"predict_fn must return [B, H, D], got shape {pred_np.shape}"
                )
            if target_np.ndim != 3:
                raise ValueError(
                    f"batch[{self.config.target_key!r}] must have shape [B, H, D], "
                    f"got {target_np.shape}"
                )
            if pred_np.shape != target_np.shape:
                raise ValueError(
                    f"prediction shape {pred_np.shape} does not match target shape {target_np.shape}"
                )

            for offset, global_idx in enumerate(batch_indices):
                predictions_by_index[global_idx] = np.asarray(pred_np[offset])
                targets_by_index[global_idx] = np.asarray(target_np[offset])
        return predictions_by_index, targets_by_index

    def _compute_from_index_maps(
        self,
        predictions_by_index: dict[int, np.ndarray],
        targets_by_index: dict[int, np.ndarray],
    ) -> dict[str, Any]:
        predictions, targets, episode_ends, intervals, episode_indices, spans = (
            self._assemble_arrays(predictions_by_index, targets_by_index)
        )

        errors, out_episode_ends, out_interval_ends = self.evaluator.compute(
            predictions,
            targets,
            episode_ends,
            intervals,
        )
        if np.isnan(errors).any():
            raise ValueError(
                "CI-MSE produced NaN errors; required validation indices may be incomplete."
            )

        return {
            **_summary(errors),
            "errors": errors,
            "episode_ends": out_episode_ends,
            "interval_ends": out_interval_ends,
            "episode_indices": episode_indices,
            "episode_spans": spans,
        }

    def _load_critical_intervals(self) -> list[list[tuple[int, int]]]:
        num_episodes = self._num_episodes()
        fps = float(getattr(self.dataset, "fps", 10))
        with open(self.intervals_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        intervals = [[] for _ in range(num_episodes)]
        for entry in data:
            ep_idx = int(entry["episode"])
            if ep_idx < 0 or ep_idx >= num_episodes:
                continue
            for interval in entry.get("intervals", []):
                start = interval.get("start")
                end = interval.get("end")
                if start is None or end is None:
                    continue
                if isinstance(start, float) or isinstance(end, float):
                    start = round(float(start) * fps)
                    end = round(float(end) * fps)
                intervals[ep_idx].append((int(start), int(end)))

        for ep_intervals in intervals:
            ep_intervals.sort(key=lambda item: item[0])
        return intervals

    def _assemble_arrays(
        self,
        predictions_by_index: dict[int, np.ndarray],
        targets_by_index: dict[int, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[tuple[int, int]]], list[int], list[tuple[int, int]]]:
        pred0 = next(iter(predictions_by_index.values()))
        H, D = pred0.shape

        prediction_blocks: list[np.ndarray] = []
        target_blocks: list[np.ndarray] = []
        episode_ends: list[int] = []
        shifted_intervals: list[list[tuple[int, int]]] = []
        episode_indices: list[int] = []
        spans: list[tuple[int, int]] = []
        total = 0

        required_set = set(predictions_by_index)
        for ep_idx, intervals in self._iter_episode_intervals():
            ep_start, ep_end = self._episode_bounds(ep_idx)
            locals_required = [
                global_idx - ep_start
                for global_idx in required_set
                if ep_start <= global_idx < ep_end
            ]
            if not locals_required:
                continue

            span_start = min(locals_required)
            span_end = max(locals_required) + 1
            block_len = span_end - span_start
            pred_block = np.full((block_len, H, D), np.nan, dtype=np.float64)
            target_block = np.full((block_len, H, D), np.nan, dtype=np.float64)

            for local_idx in locals_required:
                global_idx = ep_start + local_idx
                offset = local_idx - span_start
                pred_block[offset] = predictions_by_index[global_idx]
                target_block[offset] = targets_by_index[global_idx]

            ep_len = ep_end - ep_start
            ep_intervals: list[tuple[int, int]] = []
            for start, end in intervals:
                s = max(0, min(ep_len, int(start)))
                t = max(0, min(ep_len, int(end)))
                if t <= s:
                    continue
                ep_intervals.append((s - span_start, t - span_start))

            prediction_blocks.append(pred_block)
            target_blocks.append(target_block)
            total += block_len
            episode_ends.append(total)
            shifted_intervals.append(ep_intervals)
            episode_indices.append(ep_idx)
            spans.append((span_start, span_end))

        return (
            np.concatenate(prediction_blocks, axis=0),
            np.concatenate(target_blocks, axis=0),
            np.asarray(episode_ends, dtype=np.int64),
            shifted_intervals,
            episode_indices,
            spans,
        )

    def _iter_episode_intervals(self) -> Iterable[tuple[int, list[tuple[int, int]]]]:
        episode_filter = set(self.config.episodes) if self.config.episodes is not None else None
        for ep_idx, intervals in enumerate(self.critical_intervals):
            if episode_filter is not None and ep_idx not in episode_filter:
                continue
            if intervals:
                yield ep_idx, intervals

    def _infer_action_horizon(self) -> int | None:
        if self.config.action_horizon is not None:
            return int(self.config.action_horizon)
        for ep_idx, intervals in self._iter_episode_intervals():
            ep_start, ep_end = self._episode_bounds(ep_idx)
            ep_len = ep_end - ep_start
            for start, end in intervals:
                s = max(0, min(ep_len, int(start)))
                t = max(0, min(ep_len, int(end)))
                if t > s:
                    target = _to_numpy(self.dataset[ep_start + s][self.config.target_key])
                    if target.ndim < 2:
                        raise ValueError(
                            f"dataset target {self.config.target_key!r} must be [H, D], "
                            f"got shape {target.shape}"
                        )
                    return int(target.shape[0])
        return None

    def _lookback(self, action_horizon: int) -> int:
        cfg = self.config.ci_mse_config
        if not cfg.use_temporal_ensemble or cfg.ensemble_horizon <= 1:
            return 0
        h_end = action_horizon if cfg.horizon_end == -1 else min(action_horizon, cfg.horizon_end)
        if h_end <= cfg.horizon_start:
            return 0
        return max(0, min(cfg.ensemble_horizon - 1, action_horizon - 1 - cfg.horizon_start))

    def _episode_bounds(self, ep_idx: int) -> tuple[int, int]:
        index = self.dataset.episode_data_index
        return _as_int(index["from"][ep_idx]), _as_int(index["to"][ep_idx])

    def _num_episodes(self) -> int:
        if hasattr(self.dataset, "num_episodes"):
            return int(self.dataset.num_episodes)
        return len(self.dataset.episode_data_index["from"])

    @staticmethod
    def _chunks(values: list[int], chunk_size: int) -> Iterable[list[int]]:
        if chunk_size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(values), chunk_size):
            yield values[start : start + chunk_size]


def evaluate_lerobot_critical_interval_mse(
    dataset: Any,
    intervals_path: str | Path,
    predict_fn: Callable[[dict[str, Any]], Any],
    config: LeRobotCriticalIntervalMSEConfig | None = None,
    collate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Convenience wrapper for one-off LeRobot CI-MSE evaluation."""
    metric = LeRobotCriticalIntervalMSE(
        dataset=dataset,
        intervals_path=intervals_path,
        config=config,
        collate_fn=collate_fn,
    )
    return metric.evaluate(predict_fn)
