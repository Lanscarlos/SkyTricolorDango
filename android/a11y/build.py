"""编译读无障碍节点的小客户端：javac（照 stubs/ 编译）→ d8 → assets/a11y/skydango-a11y.jar。

用法（在仓库根目录）：python android/a11y/build.py
要 JDK（javac 在 PATH 上）和 R8 的 jar（Google Maven 的 com.android.tools:r8，含 d8）：
默认找 .pydeps/r8/r8-*.jar，或者用环境变量 R8_JAR 指定。不需要 Android SDK。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "assets" / "a11y" / "skydango-a11y.jar"


def find_r8() -> Path:
    if os.environ.get("R8_JAR"):
        return Path(os.environ["R8_JAR"])
    # worktree 里没有 .pydeps，往上找主目录的
    for base in (ROOT, *ROOT.parents):
        jars = sorted((base / ".pydeps" / "r8").glob("r8-*.jar"))
        if jars:
            return jars[-1]
    sys.exit("找不到 r8 的 jar：放到 .pydeps/r8/，或者设 R8_JAR")


def run(cmd: list[str]) -> None:
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    r8 = find_r8()
    build = ROOT / "tmp" / "a11y-build"
    shutil.rmtree(build, ignore_errors=True)
    stubs_out, classes_out = build / "stubs", build / "classes"
    stubs = [str(p) for p in (HERE / "stubs").rglob("*.java")]
    srcs = [str(p) for p in (HERE / "src").rglob("*.java")]
    run(["javac", "--release", "8", "-encoding", "UTF-8", "-nowarn", "-d", str(stubs_out), *stubs])
    run(["javac", "--release", "8", "-encoding", "UTF-8", "-cp", str(stubs_out), "-d", str(classes_out), *srcs])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    classes = [str(p) for p in classes_out.rglob("*.class")]
    run(["java", "-cp", str(r8), "com.android.tools.r8.D8", "--release", "--min-api", "26",
         "--classpath", str(stubs_out), "--output", str(OUT), *classes])
    print(f"已生成 {OUT.relative_to(ROOT)}（{OUT.stat().st_size} 字节）")


if __name__ == "__main__":
    main()
