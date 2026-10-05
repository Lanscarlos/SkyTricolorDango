"""调模型出的错：down 说明这家这次运行里还能不能用（'limit' 额度 / 余额用完、'auth' 认证失败、None 别的错）。"""
from __future__ import annotations


class ModelError(RuntimeError):
    def __init__(self, message: str, *, down: str | None = None, provider: str = "") -> None:
        super().__init__(message)
        self.down = down
        self.provider = provider


class ModelUnavailable(ModelError):
    """这个用处的主和备都不能用（调用方按"停用"处理：眼睛不看、随手记跳过、反思沿用上一份）。"""


def down_kind(exc: BaseException) -> str | None:
    """这个异常说明那家整个不能用了吗：'limit' / 'auth' / None（别的错，只认 ModelError）。"""
    return exc.down if isinstance(exc, ModelError) else None
