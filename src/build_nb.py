"""Convert percent-format scripts (notebooks/*.py, cells separated by '# %%') to executed .ipynb.

Usage: .venv\\Scripts\\python.exe src\\build_nb.py notebooks\\01_distributions.py [...]
"""
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def to_notebook(src: Path) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for block in src.read_text(encoding="utf-8").split("# %%"):
        lines = block.splitlines()
        if not lines or not "".join(lines).strip():
            continue
        header, body = lines[0].strip(), "\n".join(lines[1:]).strip("\n")
        if header.startswith("[markdown]"):
            md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
            nb.cells.append(nbformat.v4.new_markdown_cell(md))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(body))
    return nb


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        src = Path(arg).resolve()
        nb = to_notebook(src)
        ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(src.parent)}})
        out = src.with_suffix(".ipynb")
        nbformat.write(nb, out)
        print("wrote", out)
