import logging
import sys
import types

import pytest

from skydango.vision.onnxrt import open_session, providers_for, torch_device

CPU, CUDA, DML = "CPUExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider"


def test_dml_uses_directml_then_cpu_when_available():
    assert providers_for("dml", [DML, CPU]) == [DML, CPU]


def test_dml_falls_back_to_cpu_without_directml():
    assert providers_for("dml", [CUDA, CPU]) == [CPU]


def test_cuda_and_cpu_unchanged():
    assert providers_for("cuda", [CUDA, DML, CPU]) == [CUDA, CPU]
    assert providers_for("cpu", [CUDA, DML, CPU]) == [CPU]


def test_unknown_device_names_the_setting():
    with pytest.raises(ValueError, match="perception.device"):
        providers_for("directml", [CPU], what="perception.device")


class _FakeOrt(types.ModuleType):
    """装不了 onnxruntime-directml 的机器上（Linux）模拟一个有 DirectML 的 onnxruntime。"""

    class ExecutionMode:
        ORT_SEQUENTIAL = "seq"
        ORT_PARALLEL = "par"

    class SessionOptions:
        def __init__(self):
            self.enable_mem_pattern = True
            self.execution_mode = "par"

    def __init__(self, available):
        super().__init__("onnxruntime")
        self.available = available
        self.created = []

    def get_available_providers(self):
        return list(self.available)

    def InferenceSession(self, path, sess_options=None, providers=None):  # noqa: N802 - 和 onnxruntime 同名
        self.created.append((path, sess_options, providers))
        return types.SimpleNamespace(get_providers=lambda: list(providers))


def test_directml_session_turns_off_mem_pattern_and_parallel(monkeypatch):
    fake = _FakeOrt([DML, CPU])
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    open_session("m.onnx", "dml")
    _, opts, providers = fake.created[0]
    assert providers == [DML, CPU]
    assert opts.enable_mem_pattern is False and opts.execution_mode == "seq"  # DirectML 不支持这两个，开着会直接报错


def test_missing_gpu_provider_warns_with_the_right_package(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "onnxruntime", _FakeOrt([CPU]))
    with caplog.at_level(logging.WARNING):
        open_session("m.onnx", "dml")
    assert "onnxruntime-directml" in caplog.text


def test_cpu_session_does_not_warn(monkeypatch, caplog):
    fake = _FakeOrt([CPU])
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    with caplog.at_level(logging.WARNING):
        open_session("m.onnx", "cpu")
    assert caplog.text == "" and fake.created[0][2] == [CPU]


def test_torch_device_dml_runs_on_cpu_and_suggests_onnx(caplog):
    with caplog.at_level(logging.WARNING):
        assert torch_device("dml") == "cpu"
    assert ".onnx" in caplog.text
    assert torch_device("cuda") == 0 and torch_device("cpu") == "cpu"


def _tiny_yolo(path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32])
    out = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 3, 32, 32])
    model = helper.make_model(helper.make_graph([helper.make_node("Identity", ["images"], ["output0"])], "g", [inp], [out]),
                              opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))


def test_yolo_detector_on_dml_without_directml_runs_on_cpu(tmp_path, caplog):
    pytest.importorskip("onnxruntime")
    from skydango.vision.detect import OnnxYoloDetector

    _tiny_yolo(tmp_path / "y.onnx")
    with caplog.at_level(logging.WARNING):
        det = OnnxYoloDetector(tmp_path / "y.onnx", ["player"], device="dml")
    assert det.providers == [CPU] and det.imgsz == 32
    assert "onnxruntime-directml" in caplog.text


def test_perception_bench_accepts_dml(tmp_path, monkeypatch):
    from skydango import cli

    seen = []
    monkeypatch.setattr(cli, "_perception_bench", lambda cfg, args: seen.append(args.device))
    monkeypatch.chdir(tmp_path)  # 没有 config.toml：用默认配置
    cli.main(["perception", "bench", "--device", "dml", "--images", str(tmp_path)])
    assert seen == ["dml"]
