import pytest
import torch

from gnn_kt.cold_start import (
    DEFAULT_BASELINE_CONFIDENCE,
    DEFAULT_BASELINE_FORGETTING_RISK,
    DEFAULT_BASELINE_MASTERY,
    initialize_cold_start_prior,
    validate_tensor_dimensions,
)


def test_cold_start_prior_initialization() -> None:
    num_concepts = 12
    prior = initialize_cold_start_prior(num_concepts)

    assert prior["concept_count"] == num_concepts
    assert prior["is_cold_start"] is True
    assert len(prior["mastery_probabilities"]) == num_concepts
    assert len(prior["forgetting_risks"]) == num_concepts
    assert len(prior["confidence_scores"]) == num_concepts

    # Check baseline prior values
    for m in prior["mastery_probabilities"]:
        assert m == pytest.approx(DEFAULT_BASELINE_MASTERY)
    for r in prior["forgetting_risks"]:
        assert r == pytest.approx(DEFAULT_BASELINE_FORGETTING_RISK)
    for c in prior["confidence_scores"]:
        assert c == pytest.approx(DEFAULT_BASELINE_CONFIDENCE)


def test_cold_start_tensor_dimensions_and_validity() -> None:
    prior = initialize_cold_start_prior(8)
    tensor = prior["feature_tensor"]

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (8, 3)
    assert validate_tensor_dimensions(tensor, expected_features=3) is True


def test_cold_start_zero_concepts_raises_error() -> None:
    with pytest.raises(ValueError, match="concept_count must be a positive integer"):
        initialize_cold_start_prior(0)


def test_validate_tensor_dimensions_catches_nans() -> None:
    bad_tensor = torch.tensor([[0.5, float("nan"), 0.1]], dtype=torch.float32)
    assert validate_tensor_dimensions(bad_tensor) is False
