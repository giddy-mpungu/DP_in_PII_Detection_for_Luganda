"""Converts booktabs-style LaTeX tables to fully ruled (gridded) tables, the dissertation's table style:
vertical rules between all columns, a horizontal rule after every row, and a bold header row.

    gridify(tex) -> tex      every tabular environment in `tex` is converted; other text is unchanged
    python latex_grid.py <file.tex> [...]   converts files in place
"""
import re
import sys

_COL = re.compile(r"\s*(>\{(?:[^{}]|\{[^{}]*\})*\})?\s*([lcrX]|[pmb]\{(?:[^{}]|\{[^{}]*\})*\})")


def _split_spec(spec):
    spec = spec.replace("|", "").replace(" ", "")
    cols, i = [], 0
    while i < len(spec):
        if spec.startswith("@{", i):              # drop @{} inter-column overrides: rules need the default gap
            depth, j = 0, i + 1
            while True:
                depth += spec[j] == "{"
                depth -= spec[j] == "}"
                j += 1
                if depth == 0:
                    break
            i = j
            continue
        m = _COL.match(spec, i)
        if not m:
            raise ValueError(f"unparsed column spec: {spec[i:]}")
        cols.append((m.group(1) or "") + m.group(2))
        i = m.end()
    return cols


def _grid_spec(spec):
    # paragraph columns are set ragged-right: justified text in narrow cells leaves wide gaps
    cols = [c if not c.startswith(("p{", "m{", "b{")) else r">{\raggedright\arraybackslash}" + c
            for c in _split_spec(spec)]
    return "|" + "|".join(cols) + "|"


def _cells(row):
    """Split a table row on top-level & (not inside braces)."""
    out, depth, cur = [], 0, ""
    for ch in row:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "&" and depth == 0 and not cur.endswith("\\"):
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return out


def _fix_multicolumn(cell, first, bold):
    m = re.match(r"(\s*)\\multicolumn\{(\d+)\}\{([^{}]*)\}\{", cell)
    if m:
        align = m.group(3).replace("|", "") or "c"
        spec = ("|" if first else "") + align + "|"
        rest = cell[m.end():]
        return f"{m.group(1)}\\multicolumn{{{m.group(2)}}}{{{spec}}}{{" + ("\\bfseries " if bold and rest.strip() != "}" else "") + rest
    if bold and cell.strip():
        lead = cell[:len(cell) - len(cell.lstrip())]
        return f"{lead}\\bfseries {cell.lstrip()}"
    return cell


def _grid_body(body):
    body = re.sub(r"\\cmidrule(\([a-z]*\))?\{(\d+-\d+)\}", r"\\cline{\2}", body)
    # rows are separated by \\ ; rules (\toprule etc.) sit between rows
    header_zone = "\\toprule" in body and "\\midrule" in body
    parts = re.split(r"(\\\\(?:\[[^\]]*\])?)", body)
    out, in_header, seen_top = [], header_zone, False
    for k in range(0, len(parts), 2):
        row = parts[k]
        sep = parts[k + 1] if k + 1 < len(parts) else ""
        # move leading rules out of the row text
        rules = ""
        while True:
            m = re.match(r"\s*(\\toprule|\\midrule|\\bottomrule|\\hline|\\cline\{[^}]*\})\s*", row)
            if not m:
                break
            if m.group(1) == "\\toprule":
                seen_top = True
            if m.group(1) == "\\midrule" and seen_top:
                in_header = False
            rules += "" if m.group(1) in ("\\toprule", "\\midrule", "\\bottomrule", "\\hline") else m.group(1)
            row = row[m.end():]
        if rules:
            out.append(rules + "\n")
        row = row.lstrip("\n")                        # no blank lines inside a tabular
        if not sep:                                   # trailing text after the last row
            out.append(re.sub(r"\\(top|mid|bottom)rule", "", row))
            continue
        cells = _cells(row)
        cells = [_fix_multicolumn(c, i == 0, in_header) for i, c in enumerate(cells)]
        out.append("&".join(cells) + sep)
        # a full rule after every row unless the next part starts with a partial rule (\cline)
        nxt = parts[k + 2] if k + 2 < len(parts) else ""
        if not re.match(r"\s*\\cline", nxt):
            out.append(" \\hline\n")
        else:
            out.append("\n")
    return "\\hline\n" + "".join(out).lstrip("\n")


def gridify(tex):
    out, pos = [], 0
    pat = re.compile(r"\\begin\{(tabular|tabularx|longtable)\}(\{[^{}]*\})?\{")
    while True:
        m = pat.search(tex, pos)
        if not m:
            out.append(tex[pos:])
            return "".join(out)
        env = m.group(1)
        # column spec = balanced braces starting at m.end()-1
        depth, j = 0, m.end() - 1
        while True:
            depth += tex[j] == "{"
            depth -= tex[j] == "}"
            j += 1
            if depth == 0:
                break
        spec = tex[m.end():j - 1]
        end = tex.index(f"\\end{{{env}}}", j)
        body = tex[j:end]
        if "\\hline" in body and "\\toprule" not in body and "|" in spec:   # already gridded
            out.append(tex[pos:end])
            pos = end
            continue
        out.append(tex[pos:m.start()])
        out.append(tex[m.start():m.end()] + _grid_spec(spec) + "}")
        out.append("\n" + _grid_body(body).rstrip() + "\n")
        pos = end


if __name__ == "__main__":
    for path in sys.argv[1:]:
        s = open(path, encoding="utf-8").read()
        g = gridify(s)
        if g != s:
            open(path, "w", encoding="utf-8").write(g)
            print("gridded", path)
