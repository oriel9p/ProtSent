"""Shared pytest fixtures for protein pipeline tests."""

import pytest
import torch


@pytest.fixture(scope="session")
def device():
    """Return the best available compute device."""
    return "cuda" if torch.cuda.is_available() else "cpu"
