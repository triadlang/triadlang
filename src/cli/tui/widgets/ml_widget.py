from __future__ import annotations

import os
import sys
import time

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Label, RichLog, Select, Static, Switch

from cli.tui.widgets.viz_canvas import model_arch_ascii, sparkline

PY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PY_ROOT not in sys.path:
    sys.path.insert(0, PY_ROOT)

class MLWidget(Static):

    DEFAULT_CSS = """
    MLWidget {
        height: 1fr;
        overflow-y: auto;
    }
    #ml-controls {
        padding: 0 1;
        height: auto;
    }
    #ml-output {
        height: 1fr;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("f5", "train", "Train"),
        Binding("ctrl+r", "reset", "Reset"),
    ]

    def compose(self):
        with Vertical():
            yield Label("[bold cyan]🧠 ML Lab[/bold cyan]")
            with Horizontal(id="ml-controls"):
                yield Select(
                    [("XOR Classification", "xor"), ("triad Regression", "triad"), ("Sine Wave Fit", "sine")],
                    prompt="Task",
                    id="ml-task",
                    value="xor",
                )
                yield Input(placeholder="Epochs", value="100", id="ml-epochs")
                yield Input(placeholder="Learning Rate", value="0.01", id="ml-lr")
                yield Input(placeholder="Hidden Size", value="32", id="ml-hidden")
                yield Switch(id="ml-use-triad-nn", value=True)
                yield Label("[dim]Triad NN[/dim]")
                yield Button("▶ Train", variant="success", id="ml-train-btn")
                yield Button("↺ Reset", variant="default", id="ml-reset-btn")
            yield RichLog(id="ml-output", highlight=True, markup=True, wrap=False)

    def on_mount(self):
        log = self.query_one("#ml-output", RichLog)
        log.write(Panel(
            Text.from_markup(
                "[bold cyan]ML Lab[/bold cyan]\n\n"
                "Train neural networks using TriadLang's native ML runtime.\n"
                "Choose a task, set parameters, and press [bold]▶ Train[/bold] or [bold]F5[/bold].\n\n"
                "[dim]Triad NN: uses runtime.ml.nn (Module, triad, Adam)[/dim]\n"
                "[dim]Fallback: pure NumPy implementation[/dim]"
            ),
            border_style="cyan",
            title="ML Lab",
        ))

    def on_button_pressed(self, event):
        if event.button.id == "ml-train-btn":
            self.action_train()
        elif event.button.id == "ml-reset-btn":
            self._reset()

    def action_train(self):
        self._train()

    def action_reset(self):
        self._reset()

    def _reset(self):
        log = self.query_one("#ml-output", RichLog)
        log.clear()
        log.write(Text("Reset — ready for new training.", style="dim"))

    def _train(self):
        output = self.query_one("#ml-output", RichLog)
        output.clear()

        task = self.query_one("#ml-task", Select).value or "xor"
        try:
            epochs = int(self.query_one("#ml-epochs", Input).value or "100")
            lr = float(self.query_one("#ml-lr", Input).value or "0.01")
            hidden = int(self.query_one("#ml-hidden", Input).value or "32")
        except ValueError:
            output.write(Text("✗ Invalid parameters", style="red"))
            return

        use_triad = self.query_one("#ml-use-triad-nn", Switch).value

        if task == "xor":
            self._train_xor(output, epochs, lr, hidden, use_triad)
        elif task == "triad":
            self._train_triad(output, epochs, lr, hidden, use_triad)
        elif task == "sine":
            self._train_sine(output, epochs, lr, hidden, use_triad)

    def _train_xor(self, output, epochs, lr, hidden, use_triad):

        from triad import ntri as np

        output.write(Text("Task: XOR Classification", style="bold cyan"))
        output.write(Text(model_arch_ascii([
            {"name": "triad", "detail": f"2 → {hidden}"},
            {"name": "ReLU", "detail": ""},
            {"name": "triad", "detail": f"{hidden} → 1"},
            {"name": "Sigmoid", "detail": ""},
        ])))

        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
        Y = np.array([[0], [1], [1], [0]], dtype=np.float64)

        if use_triad:
            try:
                self._train_xor_triad(output, X, Y, epochs, lr, hidden)
                return
            except Exception as e:
                output.write(Text(f"Triad NN failed: {e} — falling back to NumPy", style="yellow"))

        np.random.seed(42)
        W1 = np.random.randn(2, hidden) * 0.5
        b1 = np.zeros(hidden)
        W2 = np.random.randn(hidden, 1) * 0.5
        b2 = np.zeros(1)

        losses, accs = [], []
        t0 = time.time()
        for ep in range(epochs):

            z1 = X @ W1 + b1
            a1 = np.maximum(0, z1)
            z2 = a1 @ W2 + b2
            pred = 1.0 / (1.0 + np.exp(-z2))
            loss = float(np.mean((pred - Y) ** 2))
            losses.append(loss)
            pred_bin = (pred > 0.5).astype(np.float64)
            acc = float(np.mean(pred_bin == Y))
            accs.append(acc)

            dz2 = pred - Y
            dW2 = a1.T @ dz2 / 4
            db2 = np.mean(dz2, axis=0)
            da1 = dz2 @ W2.T
            dz1 = da1 * (z1 > 0).astype(np.float64)
            dW1 = X.T @ dz1 / 4
            db1 = np.mean(dz1, axis=0)
            W1 -= lr * dW1; b1 -= lr * db1
            W2 -= lr * dW2; b2 -= lr * db2
        elapsed = time.time() - t0

        t = Table(title="XOR Results", border_style="green")
        t.add_column("Metric", style="cyan")
        t.add_column("Final", style="green", justify="right")
        t.add_column("History", style="yellow")
        t.add_row("Loss", f"{losses[-1]:.6f}", sparkline(losses, 40))
        t.add_row("Accuracy", f"{accs[-1]*100:.1f}%", sparkline(accs, 40))
        t.add_row("Time", f"{elapsed:.2f}s", "")
        output.write(t)

        if accs[-1] >= 0.99:
            output.write(Text("✓ Perfect — XOR solved!", style="bold green"))
        elif accs[-1] >= 0.75:
            output.write(Text("○ Converging — try more epochs", style="yellow"))
        else:
            output.write(Text("✗ Not converging — increase hidden size or epochs", style="red"))

    def _train_xor_triad(self, output, X, Y, epochs, lr, hidden):

        from runtime.ml.nn import Adam, ReLU, Sequential, Sigmoid, triad
        from runtime.ml.tensor import mse_loss, tensor
        from triad import ntri as np

        model = Sequential(
            triad(2, hidden),
            ReLU(),
            triad(hidden, 1),
            Sigmoid(),
        )
        opt = Adam(model.parameters(), lr=lr)

        X_t = tensor(X, requires_grad=False)
        Y_t = tensor(Y, requires_grad=False)

        losses = []
        t0 = time.time()
        for ep in range(epochs):
            opt.zero_grad()
            pred = model(X_t)
            loss = mse_loss(pred, Y_t)
            loss.backward()
            opt.step()
            losses.append(float(loss.data))
        elapsed = time.time() - t0

        pred_final = model(X_t)
        pred_np = pred_final.data
        pred_bin = (pred_np > 0.5).astype(np.float64)
        acc = float(np.mean(pred_bin == Y))

        t = Table(title="XOR Results (Triad NN)", border_style="green")
        t.add_column("Metric", style="cyan")
        t.add_column("Final", style="green", justify="right")
        t.add_column("History", style="yellow")
        t.add_row("Loss", f"{losses[-1]:.6f}", sparkline(losses, 40))
        t.add_row("Accuracy", f"{acc*100:.1f}%", "")
        t.add_row("Time", f"{elapsed:.2f}s", "")
        output.write(t)

        if acc >= 0.99:
            output.write(Text("✓ Perfect — XOR solved with Triad NN!", style="bold green"))
        else:
            output.write(Text(f"○ Accuracy: {acc*100:.1f}% — try more epochs", style="yellow"))

    def _train_triad(self, output, epochs, lr, hidden, use_triad):

        from triad import ntri as np
        np.random.seed(42)
        X = np.linspace(-3, 3, 100).reshape(-1, 1)
        Y = 2.0 * X + 1.0 + np.random.randn(100, 1) * 0.3

        w = np.random.randn(1, 1) * 0.1
        b = np.zeros(1)
        losses = []
        t0 = time.time()
        for ep in range(epochs):
            pred = X @ w + b
            loss = float(np.mean((pred - Y) ** 2))
            losses.append(loss)
            dw = X.T @ (pred - Y) / len(X)
            db = np.mean(pred - Y)
            w -= lr * dw
            b -= lr * db
        elapsed = time.time() - t0

        output.write(Text("Task: triad Regression  y = 2x + 1 + noise", style="bold cyan"))
        t = Table(title="triad Regression Results", border_style="green")
        t.add_column("Metric", style="cyan")
        t.add_column("Final", style="green", justify="right")
        t.add_row("Loss (MSE)", f"{losses[-1]:.6f}", "")
        t.add_row("Learned w", f"{float(w[0,0]):.4f} (true: 2.0)", "")
        t.add_row("Learned b", f"{float(b[0]):.4f} (true: 1.0)", "")
        t.add_row("Time", f"{elapsed:.2f}s", "")
        output.write(t)
        output.write(Text(f"Loss history: {sparkline(losses, 50)}", style="yellow"))

    def _train_sine(self, output, epochs, lr, hidden, use_triad):

        from triad import ntri as np
        np.random.seed(42)
        X = np.linspace(-np.pi, np.pi, 200).reshape(-1, 1)
        Y = np.sin(X)

        W1 = np.random.randn(1, hidden) * 0.5
        b1 = np.zeros(hidden)
        W2 = np.random.randn(hidden, 1) * 0.5
        b2 = np.zeros(1)

        losses = []
        t0 = time.time()
        for ep in range(epochs):
            z1 = X @ W1 + b1
            a1 = np.tanh(z1)
            z2 = a1 @ W2 + b2
            loss = float(np.mean((z2 - Y) ** 2))
            losses.append(loss)
            dz2 = z2 - Y
            dW2 = a1.T @ dz2 / len(X)
            db2 = np.mean(dz2, axis=0)
            da1 = dz2 @ W2.T
            dz1 = da1 * (1 - a1 ** 2)
            dW1 = X.T @ dz1 / len(X)
            db1 = np.mean(dz1, axis=0)
            W1 -= lr * dW1; b1 -= lr * db1
            W2 -= lr * dW2; b2 -= lr * db2
        elapsed = time.time() - t0

        output.write(Text("Task: Sine Wave Fit  y = sin(x)", style="bold cyan"))
        output.write(Text(model_arch_ascii([
            {"name": "triad", "detail": f"1 → {hidden}"},
            {"name": "Tanh", "detail": ""},
            {"name": "triad", "detail": f"{hidden} → 1"},
        ])))
        t = Table(title="Sine Fit Results", border_style="green")
        t.add_column("Metric", style="cyan")
        t.add_column("Final", style="green", justify="right")
        t.add_row("Loss (MSE)", f"{losses[-1]:.6f}", "")
        t.add_row("Time", f"{elapsed:.2f}s", "")
        output.write(t)
        output.write(Text(f"Loss history: {sparkline(losses, 50)}", style="yellow"))
        if losses[-1] < 0.01:
            output.write(Text("✓ Good fit!", style="bold green"))
        else:
            output.write(Text("○ Try more epochs or larger hidden size", style="yellow"))
