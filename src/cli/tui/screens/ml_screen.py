from __future__ import annotations

import os
import sys

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Label, RichLog

PY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PY_ROOT not in sys.path:
    sys.path.insert(0, PY_ROOT)

class MLScreen(Screen):
    BINDINGS = [
        Binding("f5", "train_step", "Train Step"),
    ]

    def compose(self):
        with Vertical():
            yield Label(" [bold cyan]ML Dashboard[/bold cyan]")
            with Horizontal():
                yield Input(placeholder="Epochs", value="10", id="ml-epochs")
                yield Input(placeholder="LR", value="0.001", id="ml-lr")
                yield Input(placeholder="Hidden dim", value="64", id="ml-hidden")
                yield Button("Train XOR", variant="primary", id="ml-train-btn")
            yield RichLog(id="ml-output", highlight=True, markup=True)

    def on_mount(self):
        output = self.query_one("#ml-output", RichLog)
        output.write(Panel(
            Text.from_markup(
                "[bold cyan]ML Dashboard[/bold cyan]\n\n"
                "Train a small neural network on the XOR problem.\n"
                "Watch loss decrease and accuracy increase in real time.\n\n"
                "Press [bold]F5[/bold] or click [bold]Train XOR[/bold] to start."
            ),
            border_style="cyan"
        ))

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "ml-train-btn":
            self._train_xor()

    def action_train_step(self):
        self._train_xor()

    def _train_xor(self):
        output = self.query_one("#ml-output", RichLog)
        output.clear()

        try:
            epochs = int(self.query_one("#ml-epochs", Input).value or "10")
            lr = float(self.query_one("#ml-lr", Input).value or "0.001")
            hidden = int(self.query_one("#ml-hidden", Input).value or "64")
        except ValueError:
            output.write(Text("Invalid parameters", style="red"))
            return

        from cli.tui.widgets.viz_canvas import bar, model_arch_ascii, sparkline

        arch = model_arch_ascii([
            {"name": "triad", "detail": f"2 → {hidden}"},
            {"name": "ReLU", "detail": ""},
            {"name": "triad", "detail": f"{hidden} → 1"},
        ])
        output.write(Text("Model Architecture:", style="bold cyan"))
        output.write(Text(arch))
        output.write(Text(""))

        try:
            from triad import ntri as np
            np.random.seed(42)

            W1 = np.random.randn(2, hidden) * 0.5
            b1 = np.zeros(hidden)
            W2 = np.random.randn(hidden, 1) * 0.5
            b2 = np.zeros(1)

            X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
            Y = np.array([[0], [1], [1], [0]], dtype=float)

            losses = []
            accs = []

            output.write(Text(f"Training for {epochs} epochs, lr={lr}:", style="bold"))
            output.write(Text(""))

            for epoch in range(epochs):
                z1 = X @ W1 + b1
                a1 = np.maximum(0, z1)
                z2 = a1 @ W2 + b2
                pred = 1.0 / (1.0 + np.exp(-z2))

                loss = np.mean((pred - Y) ** 2)
                losses.append(loss)

                preds_bin = (pred > 0.5).astype(float)
                acc = np.mean(preds_bin == Y)
                accs.append(acc)

                dz2 = pred - Y
                dW2 = a1.T @ dz2 / 4
                db2 = np.mean(dz2, axis=0)
                da1 = dz2 @ W2.T
                dz1 = da1 * (z1 > 0).astype(float)
                dW1 = X.T @ dz1 / 4
                db1 = np.mean(dz1, axis=0)

                W1 -= lr * dW1
                b1 -= lr * db1
                W2 -= lr * dW2
                b2 -= lr * db2

            loss_spark = sparkline(losses, width=40)
            acc_spark = sparkline(accs, width=40)

            t = Table(title="Training Results")
            t.add_column("Metric", style="cyan")
            t.add_column("Final", style="green", justify="right")
            t.add_column("History", style="yellow")
            t.add_row("Loss", f"{losses[-1]:.6f}", loss_spark)
            t.add_row("Accuracy", f"{accs[-1]*100:.1f}%", acc_spark)
            t.add_row("Epochs", str(epochs), "")
            output.write(t)

            output.write(Text(f"\n  Loss: {losses[-1]:.6f} {bar(1.0 - losses[-1], 1.0, 30)}", style="green"))
            output.write(Text(f"  Acc:  {accs[-1]*100:5.1f}%  {bar(accs[-1], 1.0, 30)}", style="green"))

            if accs[-1] >= 0.75:
                output.write(Text("\nTraining converged!", style="bold green"))
            else:
                output.write(Text("\nTry more epochs or higher learning rate.", style="yellow"))

        except Exception as e:
            output.write(Text(f"Training error: {e}", style="red"))

