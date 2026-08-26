
import os
import sys
from typing import Any

PY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PY_ROOT not in sys.path:
    sys.path.insert(0, PY_ROOT)

from cli.tui.app import TriadApp
from cli.tui.themes import ALL_THEMES, TRIAD_THEME, get_theme_by_name


def run_tui(args: list[str] | None = None) -> int:

    args = args or []
    file_path = None
    start_screen = "editor"
    theme_name = None
    no_onboarding = False

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--solver":
            start_screen = "solver"
        elif a == "--repl":
            start_screen = "repl"
        elif a == "--ml":
            start_screen = "ml"
        elif a == "--dsl":
            start_screen = "dsl"
        elif a == "--examples":
            start_screen = "examples"
        elif a == "--theme" and i + 1 < len(args):
            theme_name = args[i + 1]
            i += 1
        elif a == "--no-onboarding":
            no_onboarding = True
        elif a in ("-h", "--help"):
            print("TriadLang TUI")
            print("Usage: python -m cli.tui [FILE] [OPTIONS]")
            print()
            print("Options:")
            print("  --repl, --solver, --ml, --dsl, --examples   Start on the given tab")
            print("  --theme NAME                                Use a specific theme")
            print("  --no-onboarding                             Skip the first-time tour")
            print("  -h, --help                                  Show this help")
            print()
            print(f"Available themes: {', '.join(t.name for t in ALL_THEMES)}")
            return 0
        elif not a.startswith("-") and a.endswith(".tri"):
            file_path = os.path.abspath(a)
        i += 1

    if file_path is None:
        for a in args:
            if not a.startswith("-") and a.endswith(".tri"):
                file_path = os.path.abspath(a)
                break

    if theme_name and not get_theme_by_name(theme_name):
        print(f"⚠ Unknown theme: {theme_name}", file=sys.stderr)
        print(f"  Available: {', '.join(t.name for t in ALL_THEMES)}", file=sys.stderr)
        theme_name = None

    if theme_name:
        config_dir = os.path.expanduser("~/.triad")
        config_path = os.path.join(config_dir, "config.json")
        try:
            import json
            os.makedirs(config_dir, exist_ok=True)
            cfg = {}
            if os.path.exists(config_path):
                with open(config_path) as f:
                    cfg = json.load(f)
            settings = cfg.get("settings", {})
            settings["theme"] = theme_name
            cfg["settings"] = settings
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=2)
        except (OSError, ValueError):
            pass

    if no_onboarding:
        config_dir = os.path.expanduser("~/.triad")
        config_path = os.path.join(config_dir, "config.json")
        try:
            import json
            os.makedirs(config_dir, exist_ok=True)
            cfg = {}
            if os.path.exists(config_path):
                with open(config_path) as f:
                    cfg = json.load(f)
            cfg["onboarding_seen"] = True
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=2)
        except (OSError, ValueError):
            pass

    app = TriadApp(file_path=file_path, start_screen=start_screen)
    app.run()
    return 0

__all__ = ["run_tui", "TriadApp", "ALL_THEMES", "TRIAD_THEME"]
