"""剧本页的极简 markdown 渲染（console/static/markdown.js）：先转义再加标签，报告里的聊天原话不能变成 HTML。"""
import importlib.resources
import json
import shutil
import subprocess

import pytest

STATIC = importlib.resources.files("skydango.console") / "static"

CASES = [
    ("# 标题\n正文", "<h1>标题</h1>\n<p>正文</p>"),
    ("## 二\n### 三", "<h2>二</h2>\n<h3>三</h3>"),
    ("- a\n- b", "<ul><li>a</li><li>b</li></ul>"),
    ("1. a\n2. b", "<ol><li>a</li><li>b</li></ol>"),
    ("> 引用", "<blockquote>引用</blockquote>"),
    ("```\nx < y\n```", "<pre><code>x &lt; y</code></pre>"),
    ("有 `code` 和 **粗**", "<p>有 <code>code</code> 和 <strong>粗</strong></p>"),
    ("<script>alert(1)</script>", "<p>&lt;script&gt;alert(1)&lt;/script&gt;</p>"),
    ("**<img src=x onerror=1>**", "<p><strong>&lt;img src=x onerror=1&gt;</strong></p>"),
    ("a\nb\n\nc", "<p>a<br>b</p>\n<p>c</p>"),
]


def _run(src: str) -> str:
    node = shutil.which("node") or pytest.skip("没有 node")
    js = f"const {{renderMarkdown}}=require({json.dumps(str(STATIC / 'markdown.js'))});process.stdout.write(JSON.stringify(renderMarkdown({json.dumps(src)})))"
    out = subprocess.run([node, "-e", js], capture_output=True, text=True, encoding="utf-8", check=True).stdout
    return json.loads(out)


@pytest.mark.parametrize("src,html", CASES)
def test_render(src, html):
    assert _run(src) == html


def test_code_block_no_inline_and_unclosed():
    assert _run("```\n**x** `y`\n```") == "<pre><code>**x** `y`</code></pre>"
    assert _run("```\n<b>") == "<pre><code>&lt;b&gt;</code></pre>"  # 没收尾的代码块也只当代码
