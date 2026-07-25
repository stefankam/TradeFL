import numpy as np
import pytest

from tradefl.federation.fedavg import iid_partition_indices, sample_weighted_fedavg


def test_iid_partitions_cover_each_example_once():
    partitions = iid_partition_indices(num_examples=800, num_clients=5, seed=42)

    combined = np.concatenate(partitions)
    assert [len(partition) for partition in partitions] == [160] * 5
    assert sorted(combined.tolist()) == list(range(800))


def test_fedavg_is_weighted_by_client_examples():
    averaged = sample_weighted_fedavg(
        [
            ({"adapter": np.array([1.0, 3.0], dtype=np.float32)}, 1),
            ({"adapter": np.array([5.0, 7.0], dtype=np.float32)}, 3),
        ]
    )

    np.testing.assert_allclose(averaged["adapter"], np.array([4.0, 6.0], dtype=np.float32))


def test_fedavg_rejects_incompatible_model_tensors():
    with pytest.raises(ValueError, match="shapes do not match"):
        sample_weighted_fedavg(
            [
                ({"adapter": np.ones(2)}, 1),
                ({"adapter": np.ones(3)}, 1),
            ]
        )
