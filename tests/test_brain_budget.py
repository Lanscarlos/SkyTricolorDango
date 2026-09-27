from skydango.brain.budget import Budget
from skydango.config import BrainConfig


def test_cost_counts_cache_reads_cheaper():
    b = Budget(BrainConfig())
    usage = {"input_tokens": 1000, "cache_read_input_tokens": 20000, "cache_creation_input_tokens": None, "output_tokens": 300}
    assert abs(b.cost(usage) - ((1000 + 2000) * 2 + 300 * 10) / 1e6) < 1e-9


def test_levels_follow_last_hour_spend():
    b = Budget(BrainConfig(max_usd_per_hour=0.03, pause_usd_per_hour=0.05))
    big = {"input_tokens": 10000, "output_tokens": 0}  # $0.02
    b.record(big, 0.0)
    assert b.level(1.0) == "ok"
    b.record(big, 10.0)
    assert b.level(11.0) == "chat_only"
    b.record(big, 20.0)
    assert b.level(21.0) == "paused"
    assert b.level(3605.0) == "chat_only"  # 一小时前的那笔不算了
