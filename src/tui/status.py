from __future__ import annotations

from dataclasses import dataclass

from src.tui.renderer import clean_text


@dataclass(slots=True)
class ContextStatus:
    window: int
    window_assumed: bool = False
    tokens: int | None = None
    estimated: bool = False

    @property
    def fraction(self) -> float:
        return (self.tokens or 0) / self.window

    @property
    def style(self) -> str:
        return (
            "class:context.danger"
            if self.fraction >= 0.9
            else "class:context.warning"
            if self.fraction >= 0.8
            else "class:context"
        )

    def label(self, *, compact: bool = False) -> str:
        approximate = "~" if self.estimated else ""
        percentage = f"{self.fraction:.0%}" if self.tokens is not None else "--"
        if compact:
            return f"ctx {approximate}{percentage}" + (
                "*" if self.window_assumed else ""
            )
        filled = min(10, max(0, round(self.fraction * 10)))
        bar = "#" * filled + "-" * (10 - filled)
        used = f"{approximate}{self.tokens:,}" if self.tokens is not None else "--"
        assumed = " (window assumed)" if self.window_assumed else ""
        return f"Context [{bar}] {used} / {self.window:,} · {percentage}{assumed}"

    def toolbar(self, model: str, width: int):
        # Prefer a short label on narrow terminals; model text cannot inject controls.
        model = clean_text(model).replace("\n", " ")
        from prompt_toolkit.utils import get_cwidth

        label = self.label()
        if width < 72 or get_cwidth(label) + 3 + min(16, get_cwidth(model)) > width:
            label = self.label(compact=True)
        label = label[: max(0, width)]
        available = max(0, width - get_cwidth(label) - 3)
        if get_cwidth(model) > available:
            while model and get_cwidth(model + "…") > available:
                model = model[:-1]
            model = model + "…" if available else ""
        return [
            ("class:model", model),
            ("class:status", " · " if model else ""),
            (self.style, label),
        ]
