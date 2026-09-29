import shutil

import cv2
import numpy as np
import pytest
from conftest import scene
from panels_helpers import CARDS, ListOcr, line

from skydango import cli

DIALOG = [line("出错了", 300, 20), line("网络连接断开，请重试", 200, 90), line("确定", 330, 300)]


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """临时目录当当前目录：卡片复制一份（cut 会往里写），config.toml 指向它。"""
    cards = tmp_path / "cards"
    shutil.copytree(CARDS, cards)
    (tmp_path / "config.toml").write_text(f'[panels]\ncards_dir = "{cards.as_posix()}"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def fake_ocr(monkeypatch, *results):
    monkeypatch.setattr("skydango.vision.ocr.make_ocr", lambda engine, threads=8: ListOcr(*results))


def test_scan_directory_prints_every_card(workdir, capsys, monkeypatch):
    fake_ocr(monkeypatch, [])
    shots = workdir / "shots"
    shots.mkdir()
    for name in ("a.png", "b.png"):
        cv2.imwrite(str(shots / name), scene())
    cli.main(["panels", "scan", str(shots)])
    out = capsys.readouterr().out
    assert out.count("== ") == 2
    assert "emote_panel 动作面板（未核对）：没开" in out and "template pencil.png 没有模板图 ✗" in out
    assert "通用兜底：没认出面板" in out
    assert sorted(p.name for p in (workdir / "tmp" / "panels").iterdir()) == ["a.png", "b.png"]


def test_scan_reports_unknown_panel(workdir, capsys, monkeypatch):
    fake_ocr(monkeypatch, DIALOG)
    cv2.imwrite(str(workdir / "a.png"), scene())
    cli.main(["panels", "scan", str(workdir / "a.png")])
    assert "通用兜底：不认识的面板 「出错了」网络连接断开，请重试，按钮：确定" in capsys.readouterr().out


def test_read_prints_buttons(workdir, capsys, monkeypatch):
    fake_ocr(monkeypatch, DIALOG)
    cv2.imwrite(str(workdir / "a.png"), scene())
    cli.main(["panels", "read", str(workdir / "a.png")])
    out = capsys.readouterr().out
    assert "不认识的面板：「出错了」网络连接断开，请重试，按钮：确定" in out and "  确定  other" in out


def test_read_nothing_open(workdir, capsys, monkeypatch):
    fake_ocr(monkeypatch, [])
    cv2.imwrite(str(workdir / "a.png"), scene())
    cli.main(["panels", "read", str(workdir / "a.png")])
    assert "没有开着的面板" in capsys.readouterr().out


def test_cut_writes_template(workdir, capsys):
    img = np.zeros((1080, 1920, 3), np.uint8)
    cv2.imwrite(str(workdir / "big.png"), img)
    cli.main(["panels", "cut", str(workdir / "big.png"), "emote_panel", "pencil.png", "--roi", "0.5,0.5,0.6,0.6"])
    saved = cv2.imread(str(workdir / "cards" / "emote_panel" / "pencil.png"))
    assert saved.shape == (108, 192, 3)
    assert "panels scan" in capsys.readouterr().out


def test_cut_unknown_card(workdir):
    cv2.imwrite(str(workdir / "big.png"), np.zeros((100, 100, 3), np.uint8))
    with pytest.raises(SystemExit, match="nope"):
        cli.main(["panels", "cut", str(workdir / "big.png"), "nope", "x.png", "--roi", "0,0,0.5,0.5"])
