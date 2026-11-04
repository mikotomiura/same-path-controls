"""v7 keystroke capture on the work PC (01_実装仕様_v7キャプチャ.md, second edition; conditions
works/conditions/prototype_v7.json as amended up to 20d5cbc, of which the handoff holds a copy).

A schedule of presses is typed through SendInput and recorded by two loggers at once: the OS hook (WH_KEYBOARD_LL)
and a browser page (keydown). Only kinds and times are recorded, never the characters typed."""
