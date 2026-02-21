"""Convert README.md to README.pdf with embedded figures.

Uses only matplotlib (already a project dependency) — no external tools needed.

Usage:
    python make_pdf.py
"""
from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle

# ── page geometry (all units: data coordinates = inches) ─────────────────────
PAGE_W, PAGE_H = 8.27, 11.69   # A4 portrait
MARGIN_L = 1.0
MARGIN_R = 1.0
MARGIN_T = 0.85
MARGIN_B = 0.75
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R   # 6.27 inches

# ── typography (point sizes) ──────────────────────────────────────────────────
FS_H1    = 18
FS_H2    = 13
FS_H3    = 10.5
FS_BODY  = 9.5
FS_CODE  = 8
FS_TABLE = 8
FS_CAP   = 7.5
FS_FOOT  = 7

LEADING = 1.45   # line-height multiplier

# Approximate line heights in inches (pts × leading / 72)
def lh(pts: float) -> float:
    return pts * LEADING / 72.0

# Approximate chars per line for a given font size (assumes ~0.53 char-width ratio)
def cpl(pts: float) -> int:
    return max(40, int(CONTENT_W / (pts * 0.53 / 72.0)))


# ── basic LaTeX → readable text ───────────────────────────────────────────────
_LATEX = {
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\sigma": "σ",
    r"\mu": "μ", r"\lambda": "λ", r"\ell": "ℓ", r"\Sigma": "Σ",
    r"\sum": "Σ", r"\hat{alpha}": "α̂", r"\cdot": "·", r"\leq": "≤",
    r"\geq": "≥", r"\neq": "≠", r"\approx": "≈", r"\times": "×",
    r"\infty": "∞", r"\partial": "∂", r"\nabla": "∇",
    r"\hat{\alpha}": "α̂",
}

def _strip_inline(text: str) -> str:
    """Remove markdown inline markers, convert basic LaTeX, keep plain text."""
    # inline math  \( ... \)  and  \[ ... \]
    text = re.sub(r'\\[\(\[](.+?)\\[\)\]]', r'\1', text, flags=re.S)
    # bold + italic
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    # inline code
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # links  [label](url)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # LaTeX symbols
    for latex, uni in _LATEX.items():
        text = text.replace(latex, uni)
    # subscripts / superscripts written as _i  ^2
    text = re.sub(r'_\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\^\{([^}]+)\}', r'\1', text)
    return text.strip()


# ── block data model ──────────────────────────────────────────────────────────
@dataclass
class H1Block:      text: str
@dataclass
class H2Block:      text: str
@dataclass
class H3Block:      text: str
@dataclass
class PBlock:       text: str          # paragraph
@dataclass
class BulletBlock:  text: str
@dataclass
class CodeBlock:    lines: list[str]
@dataclass
class TableBlock:
    headers: list[str]
    rows: list[list[str]]
@dataclass
class ImageBlock:
    path: str
    caption: str = ""
@dataclass
class HRBlock:      pass

Block = Union[H1Block, H2Block, H3Block, PBlock, BulletBlock,
              CodeBlock, TableBlock, ImageBlock, HRBlock]


# ── markdown parser ────────────────────────────────────────────────────────────
def parse(md: str) -> list[Block]:
    lines = md.splitlines()
    blocks: list[Block] = []
    i = 0

    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()

        # blank line
        if not stripped:
            i += 1
            continue

        # H1
        if re.match(r'^# [^#]', raw):
            blocks.append(H1Block(_strip_inline(raw[2:])))
            i += 1
            continue

        # H2
        if re.match(r'^## [^#]', raw):
            blocks.append(H2Block(_strip_inline(raw[3:])))
            i += 1
            continue

        # H3
        if re.match(r'^### ', raw):
            blocks.append(H3Block(_strip_inline(raw[4:])))
            i += 1
            continue

        # HR
        if stripped in ('---', '***', '___') or re.match(r'^-{3,}$', stripped):
            blocks.append(HRBlock())
            i += 1
            continue

        # blockquote — accumulate consecutive > lines
        if raw.startswith('> '):
            bq_lines: list[str] = []
            while i < len(lines) and lines[i].startswith('> '):
                bq_lines.append(_strip_inline(lines[i][2:]))
                i += 1
            blocks.append(PBlock('[Note] ' + ' '.join(bq_lines)))
            continue

        # fenced code block
        if stripped.startswith('```'):
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                code_lines.append(lines[i])
                i += 1
            i += 1
            blocks.append(CodeBlock(code_lines))
            continue

        # image  ![alt](path)  optionally followed by italic caption *...*
        m = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', stripped)
        if m:
            img_path = m.group(2)
            caption  = m.group(1)
            i += 1
            # consume a following italic caption line if present
            if i < len(lines):
                cap_line = lines[i].strip()
                if cap_line.startswith('*') and cap_line.endswith('*'):
                    caption = _strip_inline(cap_line)
                    i += 1
            blocks.append(ImageBlock(path=img_path, caption=caption))
            continue

        # table (header | sep | rows)
        if '|' in raw:
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ''
            if re.match(r'^[\s|:=-]+$', next_line):
                headers = [_strip_inline(c) for c in raw.strip().strip('|').split('|')]
                i += 2  # skip separator
                rows: list[list[str]] = []
                while i < len(lines) and '|' in lines[i]:
                    cells = [_strip_inline(c) for c in lines[i].strip().strip('|').split('|')]
                    rows.append(cells)
                    i += 1
                blocks.append(TableBlock(headers=headers, rows=rows))
                continue

        # bullet / numbered list item
        m_bullet = re.match(r'^(\s*)([-*•]|\d+[.)]) (.+)', raw)
        if m_bullet:
            indent  = len(m_bullet.group(1))
            bullet_char = "  •" if indent == 0 else "    ◦"
            blocks.append(BulletBlock(f"{bullet_char}  {_strip_inline(m_bullet.group(3))}"))
            i += 1
            continue

        # regular paragraph — merge consecutive non-special lines
        para_parts: list[str] = []
        while i < len(lines):
            l = lines[i]
            if (not l.strip()
                    or l.startswith('#')
                    or l.startswith('```')
                    or l.startswith('!')
                    or l.startswith('> ')
                    or re.match(r'^(\s*)([-*•]|\d+[.)]) ', l)
                    or (i + 1 < len(lines) and re.match(r'^[\s|:=-]+$', lines[i+1].strip()) and '|' in l)
                    or re.match(r'^[-]{3,}$', l.strip())):
                break
            para_parts.append(_strip_inline(l))
            i += 1
        if para_parts:
            blocks.append(PBlock(' '.join(para_parts)))

    return blocks


# ── page renderer ──────────────────────────────────────────────────────────────
_ACCENT   = "#1a3a6e"
_ACCENT2  = "#2e5c9a"
_TEXT     = "#1a1a1a"
_SUBTEXT  = "#444444"
_CODE_BG  = "#f3f3f3"
_CODE_FG  = "#2a2a2a"
_TH_BG    = "#dce6f7"
_TR_BG    = "#f7f9fe"
_RULE_COL = "#b0b8cc"
_FOOT_COL = "#888888"
_NOTE_BG  = "#fffbe6"
_NOTE_BDR = "#d4a000"


class Renderer:
    def __init__(self, pdf: PdfPages, doc_title: str = ""):
        self.pdf = pdf
        self.doc_title = doc_title
        self.page_num = 0
        self._new_page()

    # ── page lifecycle ─────────────────────────────────────────────────────────
    def _new_page(self):
        self.fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        self.ax  = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, PAGE_W)
        self.ax.set_ylim(0, PAGE_H)
        self.ax.axis("off")
        self.y = PAGE_H - MARGIN_T
        self.page_num += 1

    def _hline(self, y: float, x0: float = MARGIN_L, x1: float = PAGE_W - MARGIN_R,
               color: str = _RULE_COL, lw: float = 0.5):
        self.ax.plot([x0, x1], [y, y], color=color, linewidth=lw,
                     transform=self.ax.transData, zorder=5)

    def _flush(self):
        """Save current page to PDF."""
        self._hline(MARGIN_B, color=_RULE_COL, lw=0.4)
        self.ax.text(
            PAGE_W / 2, MARGIN_B - 0.28,
            self.doc_title,
            ha="center", va="top", fontsize=FS_FOOT, color=_FOOT_COL,
            transform=self.ax.transData,
        )
        self.ax.text(
            PAGE_W - MARGIN_R, MARGIN_B - 0.28,
            str(self.page_num),
            ha="right", va="top", fontsize=FS_FOOT, color=_FOOT_COL,
            transform=self.ax.transData,
        )
        self.pdf.savefig(self.fig, bbox_inches="tight")
        plt.close(self.fig)

    def _ensure(self, needed: float):
        """Start a new page if *needed* inches won't fit."""
        if self.y - needed < MARGIN_B + 0.1:
            self._flush()
            self._new_page()

    def _gap(self, h: float):
        self.y -= h

    # ── block renderers ────────────────────────────────────────────────────────
    def h1(self, text: str):
        h = lh(FS_H1) + 0.3
        self._ensure(h)
        self._gap(0.22)
        # accent bar left of title
        self.ax.add_patch(Rectangle(
            (MARGIN_L - 0.12, self.y - lh(FS_H1) * 0.85),
            0.07, lh(FS_H1) * 0.9,
            facecolor=_ACCENT, transform=self.ax.transData, zorder=3,
        ))
        self.ax.text(
            MARGIN_L + 0.02, self.y, text,
            fontsize=FS_H1, fontweight="bold", color=_ACCENT,
            ha="left", va="top", transform=self.ax.transData,
        )
        self._gap(lh(FS_H1))
        self._hline(self.y, color=_ACCENT, lw=1.0)
        self._gap(0.14)

    def h2(self, text: str):
        h = lh(FS_H2) + 0.25
        self._ensure(h)
        self._gap(0.18)
        self.ax.text(
            MARGIN_L, self.y, text,
            fontsize=FS_H2, fontweight="bold", color=_ACCENT2,
            ha="left", va="top", transform=self.ax.transData,
        )
        self._gap(lh(FS_H2) + 0.02)
        self._hline(self.y, color=_ACCENT2, lw=0.6)
        self._gap(0.10)

    def h3(self, text: str):
        h = lh(FS_H3) + 0.12
        self._ensure(h)
        self._gap(0.10)
        self.ax.text(
            MARGIN_L, self.y, text,
            fontsize=FS_H3, fontweight="bold", color=_SUBTEXT,
            ha="left", va="top", transform=self.ax.transData,
        )
        self._gap(lh(FS_H3) + 0.08)

    def paragraph(self, text: str, is_note: bool = False):
        # Word-wrap
        indent = "    " if text.startswith("  •") or text.startswith("    ◦") else ""
        wrap_w = cpl(FS_BODY) - len(indent)
        wrapped_lines = textwrap.wrap(text, width=wrap_w)
        if not wrapped_lines:
            return

        total_h = len(wrapped_lines) * lh(FS_BODY) + 0.06
        if is_note:
            total_h += 0.12

        self._ensure(total_h)

        if is_note:
            self.ax.add_patch(FancyBboxPatch(
                (MARGIN_L - 0.08, self.y - total_h + 0.04),
                CONTENT_W + 0.16, total_h + 0.02,
                boxstyle="round,pad=0.04",
                facecolor=_NOTE_BG, edgecolor=_NOTE_BDR, linewidth=0.8,
                transform=self.ax.transData, zorder=0,
            ))
            self.ax.add_patch(Rectangle(
                (MARGIN_L - 0.08, self.y - total_h + 0.04),
                0.05, total_h + 0.02,
                facecolor=_NOTE_BDR,
                transform=self.ax.transData, zorder=1,
            ))
            self._gap(0.06)

        for line in wrapped_lines:
            self.ax.text(
                MARGIN_L, self.y, line,
                fontsize=FS_BODY, color=_TEXT,
                ha="left", va="top", transform=self.ax.transData,
            )
            self._gap(lh(FS_BODY))
        self._gap(0.06)

    def bullet(self, text: str):
        wrap_w = cpl(FS_BODY) - 6
        lines  = textwrap.wrap(text, width=wrap_w)
        if not lines:
            return
        total_h = len(lines) * lh(FS_BODY) + 0.03
        self._ensure(total_h)
        for line in lines:
            self.ax.text(
                MARGIN_L, self.y, line,
                fontsize=FS_BODY, color=_TEXT,
                ha="left", va="top", transform=self.ax.transData,
            )
            self._gap(lh(FS_BODY))
        self._gap(0.02)

    def code(self, code_lines: list[str]):
        if not code_lines:
            return
        pad_v = 0.10
        pad_h = 0.12
        n = len(code_lines)
        total_h = n * lh(FS_CODE) + 2 * pad_v + 0.06
        self._ensure(total_h)

        box_top = self.y - 0.04
        box_h   = n * lh(FS_CODE) + 2 * pad_v
        self.ax.add_patch(FancyBboxPatch(
            (MARGIN_L - pad_h, box_top - box_h),
            CONTENT_W + 2 * pad_h, box_h,
            boxstyle="round,pad=0.03",
            facecolor=_CODE_BG, edgecolor="#cccccc", linewidth=0.5,
            transform=self.ax.transData, zorder=0,
        ))
        self._gap(0.04 + pad_v)
        for cl in code_lines:
            self.ax.text(
                MARGIN_L + 0.04, self.y, cl,
                fontsize=FS_CODE, family="monospace", color=_CODE_FG,
                ha="left", va="top", transform=self.ax.transData, zorder=1,
            )
            self._gap(lh(FS_CODE))
        self._gap(pad_v + 0.06)

    def table(self, headers: list[str], rows: list[list[str]]):
        n_cols = max(len(headers), *(len(r) for r in rows) if rows else [1])
        col_w  = CONTENT_W / n_cols
        row_h  = lh(FS_TABLE) + 0.06
        total_h = (1 + len(rows)) * row_h + 0.12
        self._ensure(total_h)
        self._gap(0.06)

        def _draw_row_bg(y_top: float, fill: str, edge: str):
            self.ax.add_patch(Rectangle(
                (MARGIN_L - 0.04, y_top - row_h),
                CONTENT_W + 0.08, row_h,
                facecolor=fill, edgecolor=edge, linewidth=0.4,
                transform=self.ax.transData, zorder=0,
            ))

        def _draw_cells(y_top: float, cells: list[str], bold: bool = False):
            for ci, cell in enumerate(cells[:n_cols]):
                self.ax.text(
                    MARGIN_L + ci * col_w + 0.06, y_top - 0.03,
                    cell,
                    fontsize=FS_TABLE,
                    fontweight="bold" if bold else "normal",
                    color=_TEXT, ha="left", va="top",
                    transform=self.ax.transData, zorder=1,
                )

        _draw_row_bg(self.y, _TH_BG, "#99aacc")
        _draw_cells(self.y, headers, bold=True)
        self._gap(row_h)

        for ri, row in enumerate(rows):
            bg = _TR_BG if ri % 2 == 0 else "#ffffff"
            _draw_row_bg(self.y, bg, "#ccccdd")
            _draw_cells(self.y, row)
            self._gap(row_h)

        self._gap(0.08)

    def image(self, path: str, caption: str):
        img_path = Path(path)
        if not img_path.exists():
            return

        img = mpimg.imread(str(img_path))
        ih, iw = img.shape[:2]
        aspect = ih / iw
        img_h_inches = CONTENT_W * aspect
        cap_h = lh(FS_CAP) + 0.06 if caption else 0.0
        total_needed = img_h_inches + cap_h + 0.20

        # If the image won't fit even on a blank page, scale it to fit
        max_h = PAGE_H - MARGIN_T - MARGIN_B - cap_h - 0.20
        if img_h_inches > max_h:
            img_h_inches = max_h

        self._ensure(total_needed if total_needed <= PAGE_H - MARGIN_T - MARGIN_B
                     else img_h_inches + cap_h + 0.20)

        self._gap(0.14)
        # Draw a subtle border around the image
        self.ax.add_patch(Rectangle(
            (MARGIN_L - 0.02, self.y - img_h_inches - 0.04),
            CONTENT_W + 0.04, img_h_inches + 0.04,
            facecolor="white", edgecolor="#cccccc", linewidth=0.5,
            transform=self.ax.transData, zorder=0,
        ))
        # imshow via inset_axes for pixel-accurate rendering
        ax_img = self.fig.add_axes([
            MARGIN_L / PAGE_W,
            (self.y - img_h_inches) / PAGE_H,
            CONTENT_W / PAGE_W,
            img_h_inches / PAGE_H,
        ])
        ax_img.imshow(img, aspect="auto", interpolation="lanczos")
        ax_img.axis("off")

        self._gap(img_h_inches + 0.06)

        if caption:
            self.ax.text(
                PAGE_W / 2, self.y,
                caption,
                fontsize=FS_CAP, style="italic", color=_SUBTEXT,
                ha="center", va="top", wrap=True,
                transform=self.ax.transData,
            )
            self._gap(lh(FS_CAP) + 0.04)
        self._gap(0.08)

    def hr(self):
        self._ensure(0.15)
        self._gap(0.06)
        self._hline(self.y, color="#aaaaaa", lw=0.6)
        self._gap(0.10)

    def finalise(self):
        self._flush()


# ── main ───────────────────────────────────────────────────────────────────────
def build_pdf(
    readme: str = "README.md",
    output: str = "README.pdf",
) -> None:
    src = Path(readme).read_text(encoding="utf-8")
    blocks = parse(src)

    # Extract document title from first H1
    doc_title = next(
        (b.text for b in blocks if isinstance(b, H1Block)), "README"
    )

    with PdfPages(output) as pdf:
        r = Renderer(pdf, doc_title=doc_title)
        for block in blocks:
            if isinstance(block, H1Block):
                r.h1(block.text)
            elif isinstance(block, H2Block):
                r.h2(block.text)
            elif isinstance(block, H3Block):
                r.h3(block.text)
            elif isinstance(block, PBlock):
                is_note = block.text.startswith("[Note]")
                r.paragraph(block.text, is_note=is_note)
            elif isinstance(block, BulletBlock):
                r.bullet(block.text)
            elif isinstance(block, CodeBlock):
                r.code(block.lines)
            elif isinstance(block, TableBlock):
                r.table(block.headers, block.rows)
            elif isinstance(block, ImageBlock):
                r.image(block.path, block.caption)
            elif isinstance(block, HRBlock):
                r.hr()
        r.finalise()

    print(f"Saved {output}  ({Path(output).stat().st_size // 1024} KB)")


if __name__ == "__main__":
    build_pdf()
