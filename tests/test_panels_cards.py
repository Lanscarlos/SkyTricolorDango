import logging

import pytest
from panels_helpers import CARDS, write_card

from skydango.config import Config
from skydango.vision.panels import ACTIONS, CardError, load_cards

BASIC = '''
label = "演示面板"
{verified}
region = [0.1, 0.1, 0.9, 0.9]
[[features]]
kind = "template"
image = "pencil.png"
roi = [0.1, 0.1, 0.5, 0.5]
[close]
ways = ["key:18"]
'''


def basic(verified: bool = False) -> str:
    return BASIC.format(verified=f"verified = {'true' if verified else 'false'}")


def test_real_cards_load():
    cards = {c.name: c for c in load_cards(CARDS)}
    assert set(cards) == {"chat_log", "emote_panel", "wheel_editor", "friend_tree", "shared_invite"}
    assert not any(c.verified for c in cards.values())
    assert cards["chat_log"].allows == ACTIONS
    assert cards["shared_invite"].never == ("加入",) and cards["shared_invite"].texts[0].any == ("共享空间",)
    assert cards["shared_invite"].quick == ()
    assert cards["emote_panel"].close_ways == ("key:18",) and cards["emote_panel"].close_auto
    assert cards["wheel_editor"].close_ways == ("tap:0.983,0.03",)


def test_unverified_missing_template_warns(tmp_path, caplog):
    write_card(tmp_path, "demo", basic())
    with caplog.at_level(logging.WARNING):
        [card] = load_cards(tmp_path)
    assert card.features[0].template is None
    assert "demo" in caplog.text and "pencil.png" in caplog.text


def test_verified_missing_template_raises(tmp_path):
    write_card(tmp_path, "demo", basic(verified=True))
    with pytest.raises(CardError, match="demo.*pencil.png"):
        load_cards(tmp_path)


def test_card_missing_label_raises(tmp_path):
    write_card(tmp_path, "demo", basic().replace('label = "演示面板"', ""))
    with pytest.raises(CardError, match="demo.*label"):
        load_cards(tmp_path)


def test_card_unknown_key_raises(tmp_path):
    write_card(tmp_path, "demo", basic() .replace('image = "pencil.png"', 'image = "pencil.png"\ntreshold = 0.7'))
    with pytest.raises(CardError, match="treshold"):
        load_cards(tmp_path)


def test_card_bad_kind_raises(tmp_path):
    write_card(tmp_path, "demo", basic().replace('kind = "template"', 'kind = "colour"'))
    with pytest.raises(CardError, match="colour"):
        load_cards(tmp_path)


def test_card_bad_close_way_raises(tmp_path):
    write_card(tmp_path, "demo", basic().replace('"key:18"', '"kye:18"'))
    with pytest.raises(CardError, match="kye:18"):
        load_cards(tmp_path)


def test_card_bad_allows_raises(tmp_path):
    write_card(tmp_path, "demo", 'allows = ["fly"]\n' + basic())
    with pytest.raises(CardError, match="fly"):
        load_cards(tmp_path)


def test_card_toml_syntax_error_raises(tmp_path):
    write_card(tmp_path, "demo", "label = ")
    with pytest.raises(CardError, match="demo"):
        load_cards(tmp_path)


def test_underscore_dirs_skipped(tmp_path):
    write_card(tmp_path, "_common", "这不是 toml")
    write_card(tmp_path, "demo", basic(), images=("pencil.png",))
    assert [c.name for c in load_cards(tmp_path)] == ["demo"]


def test_template_loaded_as_trimmed_silhouette(tmp_path):
    write_card(tmp_path, "demo", basic(verified=True), images=("pencil.png",))
    [card] = load_cards(tmp_path)
    assert card.features[0].template.shape == (40, 40)
    assert card.verified and card.layer == 50 and card.confirm_frames == 2 and card.allows == ()


def test_panels_config_defaults():
    p = Config().panels
    assert p.enabled and p.never[:3] == ["购买", "充值", "支付"] and p.permit_window == 60.0 and p.read_ttl == 15.0
    assert p.unknown_roi == [0.15, 0.08, 0.85, 0.92] and p.retreat[0] == "关闭" and p.button_max_chars == 6
