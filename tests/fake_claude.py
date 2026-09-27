"""测试用的假 Claude Code：读 stream-json 消息，按 FAKE_CLAUDE_MODE 回应。

每次启动把参数和关心的环境变量记进 FAKE_CLAUDE_LOG（jsonl），每条收到的消息也记一行。
"""

import json
import os
import sys
import time

MODE = os.environ.get("FAKE_CLAUDE_MODE", "ok")
LOG = os.environ.get("FAKE_CLAUDE_LOG")


def emit(m):
    print(json.dumps(m, ensure_ascii=False), flush=True)


def record(entry):
    if LOG:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


args = sys.argv[1:]
resumed = "--resume" in args
session = args[args.index("--resume") + 1] if resumed else "fake-session-1"
record({
    "args": args,
    "token": os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"),
    "config_dir": os.environ.get("CLAUDE_CONFIG_DIR"),
    "api_key": os.environ.get("ANTHROPIC_API_KEY"),
})
for line in sys.stdin:
    content = json.loads(line)["message"]["content"]
    if isinstance(content, str):
        text, images = content, 0
    else:
        text = " ".join(b.get("text", "") for b in content if b.get("type") == "text")
        images = sum(b.get("type") == "image" for b in content)
    record({"message": text, "images": images})
    status = "failed" if MODE == "nomcp" else "connected"
    emit({"type": "system", "subtype": "init", "session_id": session, "mcp_servers": [{"name": "sky", "status": status}]})
    if MODE == "die" or (MODE == "die_unless_resume" and not resumed):
        sys.exit(1)
    if MODE == "hang":
        time.sleep(3600)
    if MODE == "limit":
        emit({"type": "result", "subtype": "success", "is_error": True, "result": "You've hit your limit · resets 5pm",
              "session_id": session, "num_turns": 1, "total_cost_usd": 0, "usage": {}})
        continue
    emit({"type": "assistant", "message": {"content": [{"type": "text", "text": "想：" + text[:20]}]}})
    emit({"type": "result", "subtype": "success", "is_error": False, "result": "收到：" + text, "session_id": session,
          "num_turns": 1, "total_cost_usd": 0.01, "usage": {"input_tokens": 10, "output_tokens": 2}})
