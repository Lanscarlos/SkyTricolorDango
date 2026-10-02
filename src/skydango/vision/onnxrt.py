"""onnxruntime 的公共部分：按 device 选执行后端、建会话（YOLO 检测、特征模型、动作模型共用）。

device：
- "cuda"：N 卡，要装 onnxruntime-gpu（CUDA 版本要和 torch 的对上：torch cu128 → onnxruntime-gpu 1.23.x）
- "dml"：DirectML，Windows 上的核显 / A 卡 / N 卡都能用，要装 onnxruntime-directml
  （和 onnxruntime / onnxruntime-gpu 只能装一个：都叫 onnxruntime 模块，后装的会盖掉前一个）
- "cpu"
要的 GPU 后端没装就退回 CPU 并警告。
"""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

CPU = "CPUExecutionProvider"
GPU = {"cuda": "CUDAExecutionProvider", "dml": "DmlExecutionProvider"}
PACKAGE = {"cuda": "onnxruntime-gpu（CUDA 版本要对上）", "dml": "onnxruntime-directml（先卸掉 onnxruntime）"}
DEVICES = ("cuda", "dml", "cpu")


def providers_for(device: str, available: list[str], what: str = "device") -> list[str]:
    """要的 GPU 后端在前、CPU 兜底；没装 GPU 后端就只有 CPU。"""
    if device not in DEVICES:
        raise ValueError(f"{what} 只能是 {' / '.join(DEVICES)}：{device!r}")
    wanted = [GPU[device], CPU] if device in GPU else [CPU]
    return [p for p in wanted if p in available] or [CPU]


_preloaded = False  # 这个进程里已经加载过 torch 带的 CUDA 运行库


def _torch_lib() -> str | None:
    """torch 的 lib 目录（cu128 版带着 cudart64_12 / cublas64_12 / cudnn64_9 这些 DLL）；没装 torch 返回 None。不 import torch。"""
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.origin:
        return None
    lib = os.path.join(os.path.dirname(spec.origin), "lib")
    return lib if os.path.isdir(lib) else None


def _preload_cuda(ort) -> None:
    """onnxruntime-gpu 找不到 CUDA 12 / cuDNN 9 的 DLL 时会悄悄退回 CPU：建 CUDA 会话前先从 torch 的 lib 目录加载一次。

    onnxruntime-gpu 1.30 默认是 CUDA 13 编译的，和 torch cu128 对不上，本机装的是 1.23.2（CUDA 12）；
    先 import torch 也行，但谁先谁后没保证。"""
    global _preloaded
    if _preloaded or not hasattr(ort, "preload_dlls"):
        return
    _preloaded = True
    lib = _torch_lib()
    if lib is None:
        return
    try:
        ort.preload_dlls(directory=lib)
    except Exception as exc:  # 加载不上就照常建会话：onnxruntime 自己会警告、退回 CPU
        log.debug("从 %s 预加载 CUDA 运行库失败：%s", lib, exc)


def open_session(path: str | Path, device: str, what: str = "device"):
    """建 onnxruntime 会话。调用方先自己 import onnxruntime、给出缺包时的提示。"""
    import onnxruntime as ort

    providers = providers_for(device, ort.get_available_providers(), what)
    if device == "cuda" and GPU["cuda"] in providers:
        _preload_cuda(ort)
    opts = ort.SessionOptions()
    if device == "dml" and GPU["dml"] in providers:
        # DirectML 不支持内存模式优化和并行执行，开着建会话会直接报错
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if device in GPU and GPU[device] not in providers:
        log.warning("%s = %s，但 onnxruntime 里没有 %s（要装 %s）：现在在 CPU 上跑", what, device, GPU[device], PACKAGE[device])
    return ort.InferenceSession(str(path), sess_options=opts, providers=providers)


def torch_device(device: str) -> int | str:
    """ultralytics（.pt / .engine）用的 device：torch 没有 DirectML，dml 退回 CPU。"""
    if device == "cuda":
        return 0
    if device == "dml":
        log.warning("device = dml 只对 .onnx 模型有用（torch 没有 DirectML）：.pt 模型现在在 CPU 上跑，核显上请导出成 .onnx")
    return "cpu"
