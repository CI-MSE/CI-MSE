import json
import sys
import types

import numpy as np

from val_metrics import (
    CriticalIntervalMSEConfig,
    LeRobotCriticalIntervalMSE,
    LeRobotCriticalIntervalMSEConfig,
)


class FakeLeRobotDataset:
    fps = 10

    def __init__(self, episode_lengths=(5,), target_key="action", horizon=4):
        starts = []
        ends = []
        total = 0
        for length in episode_lengths:
            starts.append(total)
            total += length
            ends.append(total)
        self.episode_data_index = {
            "from": np.array(starts),
            "to": np.array(ends),
        }
        self.num_episodes = len(episode_lengths)
        self.target_key = target_key
        self.horizon = horizon

    def __getitem__(self, idx):
        base = float(idx)
        target = np.arange(self.horizon, dtype=np.float64).reshape(self.horizon, 1) + base
        return {
            self.target_key: target,
            "index": idx,
        }


def write_intervals(tmp_path, entries):
    path = tmp_path / "intervals.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def test_required_indices_without_temporal_ensemble(tmp_path):
    intervals = write_intervals(
        tmp_path,
        [{"episode": 0, "intervals": [{"start": 1, "end": 3}]}],
    )
    metric = LeRobotCriticalIntervalMSE(
        dataset=FakeLeRobotDataset(),
        intervals_path=intervals,
        config=LeRobotCriticalIntervalMSEConfig(
            ci_mse_config=CriticalIntervalMSEConfig(
                use_temporal_ensemble=False,
                use_dtw=False,
            )
        ),
    )

    assert metric.required_indices() == [1, 2]


def test_required_indices_with_temporal_ensemble_lookback(tmp_path):
    intervals = write_intervals(
        tmp_path,
        [{"episode": 0, "intervals": [{"start": 2, "end": 4}]}],
    )
    metric = LeRobotCriticalIntervalMSE(
        dataset=FakeLeRobotDataset(horizon=4),
        intervals_path=intervals,
        config=LeRobotCriticalIntervalMSEConfig(
            ci_mse_config=CriticalIntervalMSEConfig(
                ensemble_horizon=3,
                use_dtw=False,
            )
        ),
    )

    assert metric.required_indices() == [0, 1, 2, 3]


def test_required_indices_clip_at_episode_start(tmp_path):
    intervals = write_intervals(
        tmp_path,
        [{"episode": 0, "intervals": [{"start": 0, "end": 2}]}],
    )
    metric = LeRobotCriticalIntervalMSE(
        dataset=FakeLeRobotDataset(horizon=4),
        intervals_path=intervals,
        config=LeRobotCriticalIntervalMSEConfig(
            ci_mse_config=CriticalIntervalMSEConfig(
                ensemble_horizon=4,
                use_dtw=False,
            )
        ),
    )

    assert metric.required_indices() == [0, 1]


def test_evaluate_with_custom_target_key(tmp_path):
    intervals = write_intervals(
        tmp_path,
        [{"episode": 0, "intervals": [{"start": 1, "end": 3}]}],
    )
    metric = LeRobotCriticalIntervalMSE(
        dataset=FakeLeRobotDataset(target_key="target_action"),
        intervals_path=intervals,
        config=LeRobotCriticalIntervalMSEConfig(
            ci_mse_config=CriticalIntervalMSEConfig(
                use_horizon_truncation=False,
                use_temporal_ensemble=False,
                use_dtw=False,
            ),
            target_key="target_action",
            batch_size=1,
        ),
        collate_fn=lambda samples: {
            "target_action": np.stack([sample["target_action"] for sample in samples]),
            "index": np.array([sample["index"] for sample in samples]),
        },
    )

    def predict_fn(batch):
        return batch["target_action"] + 1.0

    result = metric.evaluate(predict_fn)

    np.testing.assert_allclose(result["errors"], np.array([1.0, 1.0]))
    assert result["mean"] == 1.0
    assert result["num_timesteps"] == 2
    assert result["episode_indices"] == [0]
    assert result["episode_spans"] == [(1, 3)]


class FakeDist:
    def __init__(self, rank=0, world_size=2, gathered=None):
        self.rank = rank
        self.world_size = world_size
        self.gathered = gathered or []
        self.broadcast_payload = None

    def is_available(self):
        return True

    def is_initialized(self):
        return True

    def get_rank(self, group=None):
        return self.rank

    def get_world_size(self, group=None):
        return self.world_size

    def all_gather_object(self, gathered, payload, group=None):
        for i, item in enumerate(self.gathered):
            gathered[i] = item

    def broadcast_object_list(self, holder, src=0, group=None):
        if self.rank == src:
            self.broadcast_payload = holder[0]
        else:
            holder[0] = self.broadcast_payload


def install_fake_dist(monkeypatch, fake_dist):
    fake_torch = types.ModuleType("torch")
    fake_dist_module = types.ModuleType("torch.distributed")
    for name in (
        "is_available",
        "is_initialized",
        "get_rank",
        "get_world_size",
        "all_gather_object",
        "broadcast_object_list",
    ):
        setattr(fake_dist_module, name, getattr(fake_dist, name))
    fake_torch.distributed = fake_dist_module
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", fake_dist_module)


def test_evaluate_distributed_rank0_gathers_and_computes(tmp_path, monkeypatch):
    intervals = write_intervals(
        tmp_path,
        [{"episode": 0, "intervals": [{"start": 1, "end": 4}]}],
    )
    dataset = FakeLeRobotDataset(horizon=2)
    metric = LeRobotCriticalIntervalMSE(
        dataset=dataset,
        intervals_path=intervals,
        config=LeRobotCriticalIntervalMSEConfig(
            ci_mse_config=CriticalIntervalMSEConfig(
                use_horizon_truncation=False,
                use_temporal_ensemble=False,
                use_dtw=False,
            ),
            batch_size=2,
        ),
    )
    required = metric.required_indices()
    rank0 = required[0::2]
    rank1 = required[1::2]

    def payload(indices):
        return {
            "indices": indices,
            "predictions": [dataset[idx]["action"] + 1.0 for idx in indices],
            "targets": [dataset[idx]["action"] for idx in indices],
        }

    fake_dist = FakeDist(gathered=[payload(rank0), payload(rank1)])
    install_fake_dist(monkeypatch, fake_dist)

    result = metric.evaluate_distributed(lambda batch: batch["action"] + 1.0)

    np.testing.assert_allclose(result["errors"], np.array([1.0, 1.0, 1.0]))
    assert result["mean"] == 1.0
    assert fake_dist.broadcast_payload["mean"] == 1.0


def test_evaluate_distributed_nonzero_rank_returns_broadcast_summary(tmp_path, monkeypatch):
    intervals = write_intervals(
        tmp_path,
        [{"episode": 0, "intervals": [{"start": 1, "end": 4}]}],
    )
    dataset = FakeLeRobotDataset(horizon=2)
    metric = LeRobotCriticalIntervalMSE(
        dataset=dataset,
        intervals_path=intervals,
        config=LeRobotCriticalIntervalMSEConfig(
            ci_mse_config=CriticalIntervalMSEConfig(
                use_horizon_truncation=False,
                use_temporal_ensemble=False,
                use_dtw=False,
            )
        ),
    )
    fake_dist = FakeDist(rank=1)
    fake_dist.broadcast_payload = {
        "mean": 1.0,
        "median": 1.0,
        "q1": 1.0,
        "q99": 1.0,
        "mean_1_99": 1.0,
        "num_timesteps": 3,
    }
    install_fake_dist(monkeypatch, fake_dist)

    result = metric.evaluate_distributed(lambda batch: batch["action"] + 1.0)

    assert result == fake_dist.broadcast_payload
