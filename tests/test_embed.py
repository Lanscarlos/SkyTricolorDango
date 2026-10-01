import numpy as np
import pytest

from skydango.vision.embed import OnnxEmbedder, cosine, unit


def test_onnx_embedder_missing_file_names_the_setting(tmp_path):
    with pytest.raises(FileNotFoundError, match="appearance.model"):
        OnnxEmbedder(str(tmp_path / "x.onnx"), what="appearance.model")


def test_cosine_of_unit_vectors_and_dimension_mismatch():
    a, b = unit([1, 0, 0]), unit([1, 1, 0])
    assert cosine(a, a) == pytest.approx(1.0)
    assert cosine(a, b) == pytest.approx(2 ** -0.5)
    assert cosine(a, unit([1, 0])) == 0.0
