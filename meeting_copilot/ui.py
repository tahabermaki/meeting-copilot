"""Fenêtre flottante (PyQt6), toujours au premier plan.

Implémente l'interface `Display` de l'assistant via des signaux Qt : les appels
peuvent venir du thread asyncio, l'affichage se fait toujours dans le thread Qt.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QTextBlockFormat, QTextCursor, QTextOption
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizeGrip,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .config import UIConfig

STYLE = """
#card { background: rgba(22, 24, 30, 242); border-radius: 12px; }
QLabel { color: #e8e8ea; font-family: 'Segoe UI'; }
QPushButton { color: #9a9ca3; background: transparent; border: none; font-size: 14px; }
QPushButton:hover { color: #ffffff; }
QPushButton#help { color: #ffd27c; border: 1px solid #5a4a2a; border-radius: 6px;
                   padding: 2px 8px; font-size: 12px; }
QPushButton#help:hover { background: #3a3122; }
QTextBrowser { background: transparent; border: none; color: #ffffff; font-family: 'Segoe UI'; }
QScrollBar:vertical { background: #24262d; width: 10px; border-radius: 5px; }
QScrollBar::handle:vertical { background: #6a6e7a; border-radius: 5px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QLineEdit { background: #2a2d36; color: #ffffff; border: 1px solid #3c404b;
            border-radius: 6px; padding: 5px 8px; font-family: 'Segoe UI'; font-size: 13px; }
QLineEdit:focus { border: 1px solid #7cc4ff; }
"""


def source_label(source: str) -> str:
    if not source:
        return ""
    low = source.lower()
    if low in ("général", "general"):
        return "💡 Réponse générale"
    if low == "contexte":
        return "🧭 Contexte de réunion"
    return f"📄 {source}"


class Overlay(QWidget):
    answer_signal = pyqtSignal(str, str, str, str)  # titre, réponse, source, pied (délais)
    heard_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    help_signal = pyqtSignal()

    def __init__(self, cfg: UIConfig):
        super().__init__()
        self.cfg = cfg
        self.font_size = cfg.font_size
        self.on_help: Callable[[], None] | None = None
        self.on_typed: Callable[[str], None] | None = None
        self._drag_offset = None
        self._last_title = None

        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(320, 220)
        self.resize(cfg.width, cfg.height)

        card = QWidget(objectName="card")
        card.setStyleSheet(STYLE)

        # --- en-tête ---
        self.status = QLabel("Démarrage…")
        self.status.setStyleSheet("color:#8a8d96; font-size:11px;")
        smaller, bigger = QPushButton("A−"), QPushButton("A+")
        for b, d in ((smaller, -1), (bigger, +1)):
            b.setFixedWidth(28)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(lambda _=False, d=d: self.change_font(d))
        help_btn = QPushButton("🆘 Aide (F9)", objectName="help")
        help_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        help_btn.clicked.connect(self.request_help)
        close_btn = QPushButton("✕")
        close_btn.setFixedWidth(24)
        close_btn.clicked.connect(QApplication.quit)
        header = QHBoxLayout()
        for w, stretch in ((self.status, 1), (smaller, 0), (bigger, 0), (help_btn, 0), (close_btn, 0)):
            header.addWidget(w, stretch)

        # --- question et source ---
        self.question = QLabel("En attente d'une question…")
        self.question.setWordWrap(True)
        self.question.setStyleSheet("color:#9fa3ad; font-size:12px; font-style:italic;")
        self.source = QLabel("")
        self.source.setStyleSheet("color:#7cc4ff; font-size:11px; font-weight:600;")

        # --- réponse : défilante, mise en forme Markdown, jamais coupée horizontalement ---
        self.answer = QTextBrowser()
        self.answer.setOpenExternalLinks(False)
        self.answer.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.answer.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.answer.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.apply_font()

        # --- saisie + pied ---
        self.input = QLineEdit()
        self.input.setPlaceholderText("✍️ Écris une question ou une consigne… (Entrée)")
        self.input.returnPressed.connect(self.send_typed)
        self.heard = QLabel("")
        self.heard.setWordWrap(True)
        self.heard.setStyleSheet("color:#6b6e77; font-size:10px;")
        grip = QSizeGrip(card)
        footer = QHBoxLayout()
        footer.addWidget(self.heard, 1)
        footer.addWidget(grip, 0, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)

        inner = QVBoxLayout(card)
        inner.setContentsMargins(14, 10, 8, 8)
        inner.setSpacing(6)
        inner.addLayout(header)
        inner.addWidget(self.question)
        inner.addWidget(self.source)
        inner.addWidget(self.answer, 1)
        inner.addWidget(self.input)
        inner.addLayout(footer)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(card)

        self.answer_signal.connect(self._show_answer)
        self.heard_signal.connect(lambda t: self.heard.setText("Entendu : " + t[-160:]))
        self.status_signal.connect(self.status.setText)
        self.help_signal.connect(self.request_help)

        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.right() - cfg.width - 20, screen.top() + 20)

    # ---------------------------------------------- interface Display (thread-safe)
    def show_answer(self, title: str, body: str, source: str, footer: str) -> None:
        self.answer_signal.emit(title, body, source, footer)

    def show_status(self, message: str) -> None:
        self.status_signal.emit(message)

    def show_heard(self, text: str) -> None:
        self.heard_signal.emit(text)

    # ---------------------------------------------- actions
    def request_help(self) -> None:
        if self.on_help:
            self.on_help()
        else:
            self.status.setText("Pas encore connecté…")

    def send_typed(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        if self.on_typed:
            self.input.clear()
            self.on_typed(text)
        else:
            self.status.setText("Pas encore connecté…")

    def apply_font(self) -> None:
        self.answer.setStyleSheet(f"font-size:{self.font_size}px;")

    def change_font(self, delta: int) -> None:
        self.font_size = max(11, min(26, self.font_size + delta))
        self.apply_font()

    # ---------------------------------------------- rendu
    def _show_answer(self, title: str, body: str, source: str, footer: str) -> None:
        new_answer = title != self._last_title
        self._last_title = title
        self.question.setText(title)
        label = source_label(source)
        if footer:
            label = f"{label}   ·   {footer}" if label else footer
        self.source.setText(label)

        bar = self.answer.verticalScrollBar()
        pos = 0 if new_answer else bar.value()  # garde la position de lecture pendant l'écriture
        lines = [line for line in body.splitlines() if line.strip()]
        self.answer.setMarkdown("\n\n".join(lines))  # une ligne du modèle = une ligne affichée
        self._space_paragraphs()
        self.answer.horizontalScrollBar().setValue(0)
        if self.cfg.auto_grow:
            self._grow_to_fit()
        bar.setValue(pos)

    def _space_paragraphs(self) -> None:
        cursor = QTextCursor(self.answer.document())
        cursor.select(QTextCursor.SelectionType.Document)
        fmt = QTextBlockFormat()
        fmt.setBottomMargin(9)
        fmt.setLineHeight(130, QTextBlockFormat.LineHeightTypes.ProportionalHeight.value)
        cursor.mergeBlockFormat(fmt)

    def _grow_to_fit(self) -> None:
        doc = self.answer.document()
        doc.setTextWidth(self.answer.viewport().width())
        needed = int(doc.size().height()) + 12 - self.answer.viewport().height()
        if needed <= 0:
            return
        screen = (self.screen() or QApplication.primaryScreen()).availableGeometry()
        new_h = min(self.height() + needed, int(screen.height() * 0.85))
        if new_h > self.height():
            self.resize(self.width(), new_h)
            self.keep_on_screen()

    def keep_on_screen(self) -> None:
        screen = (self.screen() or QApplication.primaryScreen()).availableGeometry()
        g = self.frameGeometry()
        x = min(max(g.left(), screen.left()), screen.right() - g.width())
        y = min(max(g.top(), screen.top()), screen.bottom() - g.height())
        if (x, y) != (g.left(), g.top()):
            self.move(x, y)

    # ---------------------------------------------- déplacement à la souris
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self._drag_offset is not None:
            self.move(e.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, e):
        self._drag_offset = None
        self.keep_on_screen()
