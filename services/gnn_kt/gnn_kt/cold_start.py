from typing import Any

import torch

DEFAULT_BASELINE_MASTERY: float = 0.50
DEFAULT_BASELINE_FORGETTING_RISK: float = 0.00
DEFAULT_BASELINE_CONFIDENCE: float = 0.10


def initialize_cold_start_prior(concept_count: int) -> dict[str, Any]:
    """Generates a deterministic baseline prior state for a new student with zero telemetry history.

    Guarantees that tensors never contain NaNs, empty dimensions, or unhandled exceptions.
    """
    if concept_count <= 0:
        raise ValueError("concept_count must be a positive integer greater than zero.")

    # Shape: [concept_count, 3] -> (mastery_probability, forgetting_risk, confidence_score)
    feature_matrix = torch.tensor(
        [
            [
                DEFAULT_BASELINE_MASTERY,
                DEFAULT_BASELINE_FORGETTING_RISK,
                DEFAULT_BASELINE_CONFIDENCE,
            ]
        ]
        * concept_count,
        dtype=torch.float32,
    )

    return {
        "concept_count": concept_count,
        "mastery_probabilities": feature_matrix[:, 0].tolist(),
        "forgetting_risks": feature_matrix[:, 1].tolist(),
        "confidence_scores": feature_matrix[:, 2].tolist(),
        "feature_tensor": feature_matrix,
        "is_cold_start": True,
    }


def validate_tensor_dimensions(tensor: torch.Tensor, expected_features: int = 3) -> bool:
    """Validates tensor shape and confirms that no NaN or infinite values exist."""
    if tensor.dim() != 2:
        return False
    if tensor.shape[1] != expected_features:
        return False
    if torch.isnan(tensor).any().item():
        return False
    if torch.isinf(tensor).any().item():
        return False
    return True
