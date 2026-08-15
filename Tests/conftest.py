import numpy as np
import pytest

from vectordb import VectorDB


@pytest.fixture
def db():
    return VectorDB()


@pytest.fixture
def vector():
    rng = np.random.default_rng(42)
    v = rng.normal(size=384).astype(np.float32)
    return v / np.linalg.norm(v)