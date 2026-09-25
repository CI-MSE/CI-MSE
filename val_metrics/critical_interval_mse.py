import numpy as np
from dataclasses import dataclass, asdict
from scipy.interpolate import interp1d
from dtw import dtw


@dataclass
class CriticalIntervalMSEConfig:
    """All feature toggles and hyperparameters for Critical Interval MSE evaluation."""

    # Feature toggles
    use_critical_intervals: bool = True
    use_horizon_truncation: bool = True
    use_temporal_ensemble: bool = True
    use_dtw: bool = True

    # Horizon truncation: slice predictions along horizon axis to [horizon_start, horizon_end)
    horizon_start: int = 0
    horizon_end: int = -1  # -1 means use full horizon

    # Temporal ensemble window size (H_ens). 1 = no averaging.
    ensemble_horizon: int = 1

    # Interpolation factor for DTW. Factor k produces k * H' points. 1 = no interpolation.
    interpolation_factor: int = 1

    # DTW window size for Sakoe-Chiba band. None = no window.
    dtw_window_size: int = None

    def to_dict(self):
        return asdict(self)


class CriticalIntervalMSE:
    """Computes per-timestep action chunk errors within critical intervals.

    Supports configurable temporal ensemble, horizon truncation, DTW alignment,
    and interpolation. Each feature can be independently toggled on/off.
    """

    def __init__(self, config: CriticalIntervalMSEConfig):
        self.config = config

    def compute(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        episode_ends: np.ndarray,
        critical_intervals: list,
    ) -> tuple:
        """Evaluate action chunk errors across episodes and intervals.

        Args:
            predictions: [T_total, H, D] predicted action chunks.
            targets: [T_total, H, D] ground truth action chunks.
            episode_ends: [num_episodes] cumulative episode end indices into
                predictions/targets.
            critical_intervals: list (indexed by episode) of lists of
                (start, end) tuples defining critical intervals within each
                episode's local timestep range.

        Returns:
            errors: 1-D array of per-timestep scalar errors.
            out_episode_ends: 1-D array [num_episodes], cumulative indices
                into *errors*.
            out_interval_ends: 2-D array [num_episodes, max_intervals],
                per-episode cumulative indices (relative to episode start in
                *errors*). Unused slots are filled with -1.
        """
        cfg = self.config
        num_episodes = len(episode_ends)

        all_errors: list[float] = []
        out_episode_ends: list[int] = []
        all_interval_ends: list[list[int]] = []
        total_count = 0

        for ep_idx in range(num_episodes):
            ep_start = 0 if ep_idx == 0 else int(episode_ends[ep_idx - 1])
            ep_end = int(episode_ends[ep_idx])
            T_ep = ep_end - ep_start

            if cfg.use_critical_intervals:
                intervals = (
                    critical_intervals[ep_idx]
                    if ep_idx < len(critical_intervals)
                    else []
                )
                if not intervals:
                    out_episode_ends.append(total_count)
                    all_interval_ends.append([])
                    continue
            else:
                intervals = [(0, T_ep)]

            ep_preds = predictions[ep_start:ep_end]
            ep_targets = targets[ep_start:ep_end]

            if cfg.use_temporal_ensemble and cfg.ensemble_horizon > 1:
                ep_preds = self._temporal_ensemble(ep_preds)

            if cfg.use_horizon_truncation:
                ep_preds = self._truncate_horizon(ep_preds)
                ep_targets = self._truncate_horizon(ep_targets)

            ep_interval_ends: list[int] = []
            ep_count = 0

            for s, t in intervals:
                s_eval = max(0, int(s))
                t_eval = min(T_ep, int(t))

                for ts in range(s_eval, t_eval):
                    error = self._compute_chunk_error(ep_preds[ts], ep_targets[ts])
                    all_errors.append(error)

                interval_len = max(0, t_eval - s_eval)
                ep_count += interval_len
                ep_interval_ends.append(ep_count)

            total_count += ep_count
            out_episode_ends.append(total_count)
            all_interval_ends.append(ep_interval_ends)

        errors = (
            np.array(all_errors, dtype=np.float64)
            if all_errors
            else np.array([], dtype=np.float64)
        )
        episode_ends_out = np.array(out_episode_ends, dtype=np.int64)

        max_intervals = max((len(ie) for ie in all_interval_ends), default=0)
        interval_ends_out = np.full(
            (num_episodes, max_intervals), -1, dtype=np.int64
        )
        for i, ie in enumerate(all_interval_ends):
            for j, val in enumerate(ie):
                interval_ends_out[i, j] = val

        return errors, episode_ends_out, interval_ends_out

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _temporal_ensemble(self, predictions: np.ndarray) -> np.ndarray:
        """Vectorised temporal ensemble.

        a_hat[t, h] = (1/K) * sum_{m=0}^{K-1} a[t-m, h+m]
        where K = min(H_ens, H - h, t + 1).
        """
        T, H, D = predictions.shape
        H_ens = self.config.ensemble_horizon
        ensembled = np.zeros_like(predictions)
        counts = np.zeros((T, H), dtype=np.int32)

        for m in range(min(H_ens, H, T)):
            ensembled[m:, : H - m] += predictions[: T - m, m:]
            counts[m:, : H - m] += 1

        ensembled /= counts[:, :, np.newaxis]
        return ensembled

    def _truncate_horizon(self, data: np.ndarray) -> np.ndarray:
        """Slice the horizon dimension to [horizon_start, horizon_end)."""
        cfg = self.config
        h_end = data.shape[1] if cfg.horizon_end == -1 else cfg.horizon_end
        return data[:, cfg.horizon_start : h_end]

    def _compute_chunk_error(
        self, pred: np.ndarray, target: np.ndarray
    ) -> float:
        """Compare a single predicted chunk against its target."""
        if self.config.use_dtw:
            if self.config.interpolation_factor > 1:
                pred = self._interpolate(pred, self.config.interpolation_factor)
                target = self._interpolate(
                    target, self.config.interpolation_factor
                )
            window_size = (
                self.config.dtw_window_size * self.config.interpolation_factor
                if self.config.dtw_window_size is not None
                else None
            )
            return self._dtw_distance(pred, target, window_size)
        else:
            return float(np.mean((pred - target) ** 2))

    def _interpolate(self, chunk: np.ndarray, factor: int) -> np.ndarray:
        """Interpolate chunk [H, D] along the horizon axis to [H*factor, D]."""
        H = chunk.shape[0]
        x_orig = np.linspace(0, 1, H)
        x_new = np.linspace(0, 1, H * factor)
        return interp1d(x_orig, chunk, axis=0, kind="linear")(x_new)

    @staticmethod
    def _dtw_distance(seq1: np.ndarray, seq2: np.ndarray, window_size: int = None) -> float:
        """DTW normalised distance between two multi-dimensional sequences."""
        if window_size is not None:
            alignment = dtw(seq1, seq2, dist_method="sqeuclidean", distance_only=True,
            window_type='sakoechiba', window_args={'window_size': window_size})
        else:
            alignment = dtw(seq1, seq2, dist_method="sqeuclidean", distance_only=True)
        return float(alignment.normalizedDistance) / seq1.shape[-1]
