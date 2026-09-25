import numpy as np

from val_metrics import CriticalIntervalMSE, CriticalIntervalMSEConfig



def test_plain_mse_on_critical_interval():
    predictions = np.array(
        [
            [[1.0], [2.0]],
            [[2.0], [4.0]],
            [[3.0], [6.0]],
        ]
    )
    targets = np.array(
        [
            [[1.0], [2.0]],
            [[1.0], [2.0]],
            [[1.0], [2.0]],
        ]
    )
    episode_ends = np.array([3])
    critical_intervals = [[(1, 3)]]
    cfg = CriticalIntervalMSEConfig(
        use_horizon_truncation=False,
        use_temporal_ensemble=False,
        use_dtw=False,
    )

    errors, out_episode_ends, out_interval_ends = CriticalIntervalMSE(cfg).compute(
        predictions,
        targets,
        episode_ends,
        critical_intervals,
    )

    np.testing.assert_allclose(errors, np.array([2.5, 10.0]))
    np.testing.assert_array_equal(out_episode_ends, np.array([2]))
    np.testing.assert_array_equal(out_interval_ends, np.array([[2]]))
