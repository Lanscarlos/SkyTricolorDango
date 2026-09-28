"""YOLO 训练环境和缓存都放进仓库（.pydeps/、.cache/），不写 C 盘。

`python -m skydango.cachedirs` 在用户 site-packages 里写 skydango-pydeps.pth：
- 第一行把仓库的 .pydeps/（torch、ultralytics 等，装法见 CLAUDE.md「环境」）加进 sys.path
- 第二行每次 Python 启动时执行：当前目录在仓库里，就把 DIRS 里的缓存目录指到 .cache/ 下（已经设过的不覆盖）
.pth 自带全部逻辑、不引用仓库里的文件 —— 工作区切到还没有本文件的分支也照样生效。
"""

import os
import site
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PTH_NAME = "skydango-pydeps.pth"
DIRS = {
    "PIP_CACHE_DIR": "pip",
    "YOLO_CONFIG_DIR": "ultralytics",  # 设置文件、字体（Arial.ttf 等）
    "TORCH_HOME": "torch",  # torch.hub / torchvision 预训练权重
    "MPLCONFIGDIR": "matplotlib",  # 训练画曲线时的字体缓存
    "CUDA_CACHE_PATH": "nvidia",  # CUDA 内核 JIT 缓存
    "HF_HOME": "huggingface",
}


def pth_text(root: Path = ROOT) -> str:
    root = Path(root).resolve()
    pairs = tuple((key, str(root / ".cache" / sub)) for key, sub in DIRS.items())
    line = (
        f"import os; r = os.path.normcase({str(root)!r}); c = os.path.normcase(os.getcwd()); "
        f"(c == r or c.startswith(r + os.sep)) and [os.environ.setdefault(k, v) for k, v in {pairs!r}]"
    )
    return f"{root / '.pydeps'}\n{line}\n"


def install(root: Path = ROOT) -> Path:
    path = Path(site.getusersitepackages()) / PTH_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pth_text(root), encoding="utf-8")
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="写 skydango-pydeps.pth：.pydeps/ 加进 sys.path，在仓库里时缓存放 .cache/")
    parser.add_argument("--root", type=Path, default=ROOT, help="仓库目录（默认本文件所在的仓库）")
    print(f"已写入 {install(parser.parse_args().root)}")
