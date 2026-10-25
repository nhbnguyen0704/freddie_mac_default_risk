from datetime import datetime
from pathlib import Path
from typing import List, Sequence
import pandas as pd
from tabulate import tabulate

def md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}", index: bool = False, na: str = '--',
             numparse: bool = True) -> str:
    """A pipe Markdown table of df: floats by floatfmt, ints with thousands separators, missing as na, numeric
    columns right-aligned. A bare | in a cell is escaped so it cannot end the cell; numparse=False keeps cells
    exactly as formatted (signs, trailing zeros) instead of letting tabulate re-parse them."""
    frame = df.reset_index() if index else df
    if frame.empty:
        return "_(no rows)_"

    def cell(v) -> str:
        if pd.isna(v):
            return na
        if isinstance(v, float):
            return floatfmt.format(v)
        if isinstance(v, (int,)) and not isinstance(v, bool):
            return f"{v:,}"
        return str(v)

    def esc(s: str) -> str:
        return s.replace('|', r'\|')

    head = [esc(str(c)) for c in frame.columns]
    rows = [[esc(cell(v)) for v in row] for row in frame.itertuples(index=False)]
    numeric = [pd.api.types.is_numeric_dtype(frame[c]) for c in frame.columns]
    colalign = ['right' if n else 'left' for n in numeric]
    return tabulate(rows, headers=head, tablefmt='pipe', colalign=colalign, disable_numparse=not numparse)

def write_report(path: Path, title: str, source: str, sections: Sequence[tuple], preamble: str = "") -> Path:
    """Write a Markdown report: title, generation stamp and source, optional preamble, then one section per
    (heading, body)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out: List[str] = [f"# {title}", ""]
    out += [f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} "
            f"by `{source}`._", ""]
    if preamble:
        out += [preamble.strip(), ""]
    for heading, body in sections:
        if heading:
            out += [f"## {heading}", ""]
        out += [body.strip(), ""]
    path.write_text("\n".join(out).rstrip() + "\n")
    return path