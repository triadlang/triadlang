from __future__ import annotations

from dataclasses import dataclass

from rich.syntax import Syntax
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Label, Static


@dataclass
class Snippet:

    id: str
    title: str
    description: str
    category: str
    code: str
    icon: str = "📄"

SNIPPETS: list[Snippet] = [

    Snippet(
        "hello",
        "Hello World",
        "Classic first program",
        "Basics",
        '// Hello World in TriadLang\nprint("Hello, World!");',
        "👋",
    ),
    Snippet(
        "variable",
        "Variable Declaration",
        "Mutable variable",
        "Basics",
        'let name = "TriadLang";\nlet version = 1.0;\nprint(name, version);',
        "📦",
    ),
    Snippet(
        "function",
        "Function Definition",
        "fn with parameters and return",
        "Basics",
        '''fn add(a, b) {
    return a + b;
}
print(add(2, 3));  // 5''',
        "ƒ",
    ),
    Snippet(
        "loop-for",
        "For Loop",
        "Iterate over range or list",
        "Basics",
        '''for i in range(5) {
    print(i);
}''',
        "🔁",
    ),
    Snippet(
        "loop-while",
        "While Loop",
        "Conditional loop",
        "Basics",
        '''let x = 0;
while x < 10 {
    x = x + 1;
}
print(x);''',
        "🔄",
    ),
    Snippet(
        "if-else",
        "If / Else",
        "Conditional execution",
        "Basics",
        '''let x = 42;
if x > 50 {
    print("big");
} else if x > 25 {
    print("medium");
} else {
    print("small");
}''',
        "🔀",
    ),
    Snippet(
        "match",
        "Match Statement",
        "Pattern matching (like switch)",
        "Basics",
        '''let x = 2;
match x {
    case 1 => { print("one"); }
    case 2 => { print("two"); }
    else => { print("other"); }
}''',
        "🎯",
    ),

    Snippet(
        "list",
        "List / Array",
        "Ordered collection",
        "Data",
        '''let xs = [1, 2, 3, 4, 5];
xs.push(6);
print(len(xs));   // 6
print(xs[0]);     // 1''',
        "📋",
    ),
    Snippet(
        "dict",
        "Dictionary / Map",
        "Key-value store",
        "Data",
        '''let m = {"a": 1, "b": 2, "c": 3};
print(m["a"]);   // 1
m["d"] = 4;''',
        "🗂",
    ),
    Snippet(
        "tuple",
        "Tuple",
        "Immutable pair/sequence",
        "Data",
        '''let t = (1, 2, 3);
let (a, b, c) = t;
print(a, b, c);''',
        "📦",
    ),

    Snippet(
        "class",
        "Class Definition",
        "Object with fields and methods",
        "OOP",
        '''class Point {
    x;
    y;
    fn describe(self) {
        return self.x + self.y;
    }
}
let p = Point(3, 4);
print(p.describe());  // 7''',
        "🏛",
    ),
    Snippet(
        "inheritance",
        "Inheritance",
        "Extend a parent class",
        "OOP",
        '''class Animal {
    name;
    fn speak(self) { return self.name + " speaks"; }
}
class Dog : Animal {
    breed;
}
let d = Dog("Rex", "Lab");
print(d.speak());''',
        "🧬",
    ),

    Snippet(
        "try-catch",
        "Try / Catch",
        "Error handling",
        "Errors",
        '''try {
    throw "oops";
} catch e {
    print("Caught:", e);
} finally {
    print("done");
}''',
        "🛡",
    ),
    Snippet(
        "async-fn",
        "Async Function",
        "Async function declaration",
        "Async",
        '''async fn fetch(url) {
    return url;
}
let result = fetch("https://example.com");
print(result);''',
        "⚡",
    ),
    Snippet(
        "generator",
        "Generator (yield)",
        "Lazy sequence with yield",
        "Async",
        '''fn gen(n) {
    for i in range(n) {
        yield i * 2;
    }
}
for x in gen(5) {
    print(x);
}''',
        "🌊",
    ),

    Snippet(
        "solver-basic",
        "Basic Solver",
        "Single-region solver",
        "Physics",
        '''reg crystal {
    N   128
    T   18.0
    L   32.0
    dt  0.005
    Lambda 0.5
    Gamma 0.01
}''',
        "🔬",
    ),
    Snippet(
        "solver-coupled",
        "Coupled Pair",
        "Two coupled regions",
        "Physics",
        '''reg A { N 128  T 18.0  L 32.0  dt 0.005 }
reg B { N 128  T 18.0  L 32.0  dt 0.005 }
couple A B { kappa -2.5 }''',
        "🔗",
    ),

    Snippet(
        "ml-xor",
        "ML: XOR Training",
        "Train XOR with the ML runtime",
        "ML",
        '''import runtime.ml.tensor as T
from runtime.ml.nn import triad, ReLU, Sequential, Adam, mse_loss

let model = Sequential(
    triad(2, 8),
    ReLU(),
    triad(8, 1),
)
let opt = Adam(model.parameters(), 0.01)
let X = T.tensor([[0,0],[0,1],[1,0],[1,1]], false)
let Y = T.tensor([[0],[1],[1],[0]], false)

for epoch in range(500) {
    opt.zero_grad()
    let pred = model(X)
    let loss = mse_loss(pred, Y)
    loss.backward()
    opt.step()
}
print("Done")''',
        "🧠",
    ),
]

class SnippetSelected(Message):

    def __init__(self, snippet: Snippet):
        super().__init__()
        self.snippet = snippet

class SnippetsWidget(Static):

    DEFAULT_CSS = """
    SnippetsWidget {
        height: 1fr;
        overflow-y: auto;
    }
    #snippets-list {
        height: 1fr;
        padding: 0 1;
    }
    #snippets-list Button {
        width: 1fr;
        margin: 0 0 1 0;
    }
    #snippet-preview {
        height: 12;
        border: round $panel;
        background: $background;
        margin: 1;
    }
    """

    BINDINGS = [
        Binding("enter", "insert_snippet", "Insert"),
        Binding("escape", "app.pop_screen", "Back"),
    ]

    def __init__(self, **kw):
        super().__init__(**kw)
        self._snippets = SNIPPETS
        self._selected: Snippet | None = None

    def compose(self):
        with Vertical():
            yield Label("[bold cyan]📋 Snippets[/bold cyan]")
            yield Static(
                "[dim]Pick a snippet to insert into the editor[/dim]",
                id="snippets-hint",
            )
            with Vertical(id="snippets-list"):
                for snip in self._snippets:
                    yield Button(
                        f"{snip.icon}  {snip.title}  [dim]— {snip.description}[/dim]",
                        id=f"snip-{snip.id}",
                        variant="default",
                    )
            yield Static(
                "[dim]Preview will appear here[/dim]",
                id="snippet-preview",
            )
            with Horizontal():
                yield Button("📋 Insert in Editor", variant="success", id="snip-insert-btn")
                yield Button("📋 Copy to Clipboard", variant="primary", id="snip-copy-btn")
                yield Button("❌ Close", variant="error", id="snip-close-btn")

    def on_mount(self) -> None:
        if self._snippets:
            self._selected = self._snippets[0]
            self._update_preview()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if not event.button.id:
            return
        bid = event.button.id
        if bid.startswith("snip-"):
            snip_id = bid[5:]

            if snip_id in ("insert-btn", "copy-btn", "close-btn"):
                if snip_id == "insert-btn":
                    self.action_insert_snippet()
                elif snip_id == "copy-btn":
                    self._copy_to_clipboard()
                elif snip_id == "close-btn":
                    self.app.pop_screen()
                return

            snip = next((s for s in self._snippets if s.id == snip_id), None)
            if snip:
                self._selected = snip
                self._update_preview()

    def action_insert_snippet(self) -> None:

        if self._selected:
            self.post_message(SnippetSelected(self._selected))

    def _update_preview(self) -> None:
        if not self._selected:
            return
        preview = self.query_one("#snippet-preview", Static)
        code = self._selected.code
        preview.update(Syntax(
            code, "triad", theme="monokai",
            line_numbers=True, word_wrap=True,
        ))

    def _copy_to_clipboard(self) -> None:

        if not self._selected:
            return
        try:
            import pyperclip
            pyperclip.copy(self._selected.code)
            if hasattr(self.app, "_output"):
                self.app._output("✓ Snippet copied to clipboard", "green")
        except (ImportError, AttributeError, RuntimeError):
            if hasattr(self.app, "_output"):
                self.app._output(
                    "⚠ Clipboard unavailable — snippet in preview above",
                    "yellow",
                )

