import importlib

import manimgx as m
import welch_style as ws


def test_palette_values():
    assert ws.CHILL_BROWN == '#948979'
    assert ws.YELLOW == '#ffd35a'
    assert ws.BLUE == '#65c8d0'
    assert ws.CHILL_GREEN == '#6c946f'
    assert ws.CHILL_BLUE == '#3d5c6f'
    assert ws.YELLOW_FADE == '#7f6a2d'
    assert ws.FRESH_TAN == '#dfd0b9'
    assert ws.GREEN == '#00a14b'
    assert ws.CYAN == '#00ffff'
    assert ws.MAGENTA == '#ff00ff'


def test_sizes_and_font():
    assert ws.FONT == 'Myriad Pro'
    assert ws.BACKGROUND == '#000000'
    assert (ws.FONT_SIZE, ws.STROKE_WIDTH, ws.THIN_STROKE) == (24, 3, 2)


def test_text_default_font_applied_and_overridable():
    # set_default wraps the constructor; an explicit font must still be accepted.
    assert '_original__init__' in m.Text.__dict__
    m.Text('x')
    m.Text('x', font='Georgia')


def test_background_set():
    assert m.ManimColor(m.config.background_color) == m.ManimColor('#000000')


def test_font_installed_matches_filenames(tmp_path):
    (tmp_path / 'MyriadPro-Regular.otf').touch()
    (tmp_path / 'notes.txt').touch()
    assert ws.font_installed('Myriad Pro', dirs=[tmp_path])
    assert not ws.font_installed('Minion Pro', dirs=[tmp_path])
    assert not ws.font_installed('notes', dirs=[tmp_path])


def test_reimport_warns_once(capsys):
    importlib.import_module('welch_style')
    assert 'welch_style:' not in capsys.readouterr().out


def test_star_import_exports_only_style_names():
    names = set(ws.__all__)
    assert {'CHILL_BROWN', 'FONT', 'FONT_SIZE', 'font_installed'} <= names
    assert names.isdisjoint({'Text', 'm', 'Path', 'BLACK'})
