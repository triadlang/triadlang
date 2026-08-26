from __future__ import annotations

from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Label, RichLog, Static


class OnboardingStep:

    def __init__(self, title: str, description: str, code: str = "", tip: str = ""):
        self.title = title
        self.description = description
        self.code = code
        self.tip = tip

ONBOARDING_STEPS: list[OnboardingStep] = [
    OnboardingStep(
        "Welcome to TriadLang",
        "TriadLang is a programming language whose runtime is a  field Triad. "
        "It combines the simplicity of Python-like syntax with built-in physics, ML, and quantum "
        "computing primitives. This short tour will teach you the basics.",
        tip="Press → for next, ← for previous, Esc to skip",
    ),
    OnboardingStep(
        "1. Hello World",
        "Every language starts here. TriadLang uses C-like braces and semicolons, but the rest "
        "feels familiar. Use print() to output to the console.",
        code='// This is a comment\nprint("Hello, TriadLang!");\nprint(2 + 2);',
        tip="Click 'Run in REPL' to try it",
    ),
    OnboardingStep(
        "2. Variables and Types",
        "Use 'let' to declare variables. TriadLang uses type inference — you rarely need to "
        "declare types explicitly. Strings, numbers, and booleans are first-class.",
        code='''let name = "Triad";
let version = 1.0;
let is_cool = true;
let numbers = [1, 2, 3];
print(name, version, is_cool, numbers);''',
        tip="Numbers, strings, booleans, lists, dicts",
    ),
    OnboardingStep(
        "3. Functions",
        "Define functions with 'fn'. They support default arguments, recursion, and can return "
        "any type. Use 'return' to exit early (or omit for implicit last-expression return).",
        code='''fn fib(n) {
    if n <= 1 { return n; }
    return fib(n - 1) + fib(n - 2);
}
print(fib(10));  // 55''',
        tip="TriadLang is multi-paradigm: OOP, functional, async",
    ),
    OnboardingStep(
        "4. Control Flow",
        "You have if/else, for-in, while, and pattern matching with match/case. "
        "The 'in' keyword works for ranges and iterables.",
        code='''for i in range(5) {
    if i % 2 == 0 {
        print(i, "is even");
    } else {
        print(i, "is odd");
    }
}''',
        tip="range(5) → 0,1,2,3,4",
    ),
    OnboardingStep(
        "5. Classes and OOP",
        "TriadLang supports classes with fields, methods, and inheritance. "
        "Use 'self' to refer to the instance.",
        code='''class Animal {
    name;
    fn speak(self) {
        return self.name + " makes a sound";
    }
}
class Dog : Animal {
    fn speak(self) {
        return self.name + " barks";
    }
}
let d = Dog("Rex");
print(d.speak());  // Rex barks''',
        tip="Methods take 'self' explicitly",
    ),
    OnboardingStep(
        "6. Physics: Solvers",
        "The unique part: TriadLang has a built-in field-Triad solver. Define regions and "
        "couplings declaratively. Press F5 to run or visit the Solver tab.",
        code='''reg crystal {
    N   128
    T   18.0
    L   32.0
    dt  0.005
    Lambda 0.5
    Gamma  0.01
}''',
        tip="Try the Solver tab for live field visualization",
    ),
    OnboardingStep(
        "7. ML Built-in",
        "Train neural networks directly with 'import triad.nn'. The ML runtime is triady native — "
        "no PyTorch dependency. Triad tensors support autograd.",
        code='''import runtime.ml.tensor as T
from runtime.ml.nn import triad, ReLU, Sequential, Adam, mse_loss

let model = Sequential(triad(2, 8), ReLU(), triad(8, 1))
print(model)''',
        tip="Visit the ML Lab tab to try it interactively",
    ),
    OnboardingStep(
        "8. Errors are Friendly",
        "TriadLang has detailed error messages with file location, line, and column. "
        "Try writing invalid code to see the diagnostics in the Errors tab.",
        code='let x = ;  // syntax error',
        tip="Errors appear in the bottom Errors tab",
    ),
    OnboardingStep(
        "9. Where to Go Next",
        "You're ready! Try these next:\n\n"
        "  • Press F5 in the editor to run your code\n"
        "  • Open an example from the Examples tab (Ctrl+X)\n"
        "  • Try a snippet from Ctrl+Shift+S\n"
        "  • Customize the theme with Ctrl+G (Settings)\n"
        "  • Press Ctrl+K anytime for the Command Palette\n"
        "  • Press Ctrl+/ for a vim-style command bar\n\n"
        "Have fun!",
        tip="Welcome to TriadLang ⚡",
    ),
]

class OnboardingScreen(Screen):

    DEFAULT_CSS = """
    OnboardingScreen {
        align: center middle;
        background: $surface;
    }
    #onboard-container {
        width: 90%;
        max-width: 120;
        height: 90%;
        border: thick $primary;
        background: $panel;
    }
    #onboard-title {
        height: auto;
        padding: 1 2;
        background: $primary;
        color: $background;
    }
    #onboard-progress {
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 2;
    }
    #onboard-content {
        height: 1fr;
        padding: 1 2;
    }
    #onboard-nav {
        height: auto;
        padding: 1 2;
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("escape", "close", "Skip Tour"),
        Binding("right", "next", "Next"),
        Binding("left", "prev", "Previous"),
        Binding("space", "next", "Next"),
    ]

    def __init__(self, **kw):
        super().__init__(**kw)
        self._step = 0

    def compose(self):
        with Vertical(id="onboard-container"):
            yield Label("⚡ TriadLang Tour", id="onboard-title")
            yield Static("Step 1 / 10", id="onboard-progress")
            yield RichLog(id="onboard-content", highlight=True, markup=True, wrap=True)
            with Horizontal(id="onboard-nav"):
                yield Button("← Previous", id="onboard-prev", variant="default")
                yield Button("Next →", id="onboard-next", variant="primary")
                yield Button("Skip Tour", id="onboard-skip", variant="error")
                yield Button("📋 Copy Code", id="onboard-copy", variant="default")
                yield Button("▶ Run in REPL", id="onboard-run", variant="success")

    def on_mount(self) -> None:
        self._render()

    def _render(self) -> None:
        step = ONBOARDING_STEPS[self._step]
        log = self.query_one("#onboard-content", RichLog)
        log.clear()

        log.write(Panel(
            Text(step.title, style="bold cyan"),
            border_style="cyan",
        ))
        log.write(Text(""))

        log.write(Text(step.description, style="white"))
        log.write(Text(""))

        if step.code:
            log.write(Syntax(step.code, "triad", theme="monokai", line_numbers=True))
            log.write(Text(""))

        if step.tip:
            log.write(Panel(
                Text(f"💡 {step.tip}", style="yellow"),
                border_style="yellow",
            ))

        self.query_one("#onboard-progress", Static).update(
            f"Step {self._step + 1} / {len(ONBOARDING_STEPS)}"
        )

        prev_btn = self.query_one("#onboard-prev", Button)
        prev_btn.disabled = (self._step == 0)

    def action_next(self) -> None:
        if self._step < len(ONBOARDING_STEPS) - 1:
            self._step += 1
            self._render()

    def action_prev(self) -> None:
        if self._step > 0:
            self._step -= 1
            self._render()

    def action_close(self) -> None:
        self.dismiss()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "onboard-next":
            self.action_next()
        elif bid == "onboard-prev":
            self.action_prev()
        elif bid == "onboard-skip":
            self.action_close()
        elif bid == "onboard-copy":
            self._copy_code()
        elif bid == "onboard-run":
            self._run_in_repl()

    def _copy_code(self) -> None:
        step = ONBOARDING_STEPS[self._step]
        if not step.code:
            return
        try:
            import pyperclip
            pyperclip.copy(step.code)
        except (ImportError, AttributeError):
            pass
        if hasattr(self.app, "_output"):
            self.app._output("✓ Step code copied", "green")

    def _run_in_repl(self) -> None:
        step = ONBOARDING_STEPS[self._step]
        if not step.code:
            return
        if hasattr(self.app, "_run_code"):
            self.app._run_code(step.code, "onboarding")
        if hasattr(self.app, "action_show_tab"):
            self.app.action_show_tab("repl")
        self.action_close()

