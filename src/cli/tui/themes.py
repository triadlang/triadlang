from textual.theme import Theme

TRIAD_THEME = Theme(
    name="triad",
    primary="#00BCD4",
    secondary="#7C4DFF",
    accent="#FF6D00",
    warning="#FFD600",
    error="#FF1744",
    success="#00E676",
    surface="#1A1A2E",
    background="#0D0D1A",
    panel="#16213E",
    dark=True,
)

CATPPUCCIN_MOCHA = Theme(
    name="catppuccin-mocha",
    primary="#89b4fa",
    secondary="#cba6f7",
    accent="#fab387",
    warning="#f9e2af",
    error="#f38ba8",
    success="#a6e3a1",
    surface="#1e1e2e",
    background="#11111b",
    panel="#313244",
    dark=True,
)

TOKYO_NIGHT = Theme(
    name="tokyo-night",
    primary="#7aa2f7",
    secondary="#bb9af7",
    accent="#ff9e64",
    warning="#e0af68",
    error="#f7768e",
    success="#9ece6a",
    surface="#1f2335",
    background="#16161e",
    panel="#24283b",
    dark=True,
)

GRUVBOX_DARK = Theme(
    name="gruvbox-dark",
    primary="#83a598",
    secondary="#d3869b",
    accent="#fe8019",
    warning="#fabd2f",
    error="#fb4934",
    success="#b8bb26",
    surface="#3c3836",
    background="#282828",
    panel="#504945",
    dark=True,
)

NORD = Theme(
    name="nord",
    primary="#88c0d0",
    secondary="#b48ead",
    accent="#ebcb8b",
    warning="#ebcb8b",
    error="#bf616a",
    success="#a3be8c",
    surface="#3b4252",
    background="#2e3440",
    panel="#434c5e",
    dark=True,
)

SOLARIZED_DARK = Theme(
    name="solarized-dark",
    primary="#268bd2",
    secondary="#6c71c4",
    accent="#cb4b16",
    warning="#b58900",
    error="#dc322f",
    success="#859900",
    surface="#073642",
    background="#002b36",
    panel="#586e75",
    dark=True,
)

TRIAD_LIGHT = Theme(
    name="triad-light",
    primary="#0097A7",
    secondary="#651FFF",
    accent="#FF6D00",
    warning="#FFC107",
    error="#D50000",
    success="#00C853",
    surface="#FAFAFA",
    background="#FFFFFF",
    panel="#ECEFF1",
    dark=False,
)

ALL_THEMES = [
    TRIAD_THEME,
    CATPPUCCIN_MOCHA,
    TOKYO_NIGHT,
    GRUVBOX_DARK,
    NORD,
    SOLARIZED_DARK,
    TRIAD_LIGHT,
]

DARK_THEMES = [t for t in ALL_THEMES if t.dark]
LIGHT_THEMES = [t for t in ALL_THEMES if not t.dark]

def get_theme_by_name(name: str) -> Theme | None:

    for theme in ALL_THEMES:
        if theme.name == name:
            return theme
    return None

