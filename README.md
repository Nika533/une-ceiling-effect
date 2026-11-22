# Ceiling effects in Georgia's Unified National Exams

Code for the paper *How Georgia's Unified National Mathematics Exam Struggles to Distinguish Its Strongest Students* (Nikoloz Melikishvili, 2026).

The analysis uses public data from Georgia's National Assessment and Examinations Center (NAEC), 2020–2026: the equating tables of every exam version, the score lists of all test-takers (2024–2026) and the lists of admitted students. It measures ceilings in mathematics and other subjects, the measurement error of top scores, and how stable admission to Free University's Mathematics and Computer Science program (MACS) and two other programs would be on a simulated retake.

## Repository structure
- `src/` – data loading (`load.py`), recovery of raw scores from scaled scores (`scores.py`), the true-score retake simulation (`simulate.py`), parsers for the NAEC PDF and HTML files, and the notebook builder (`build_nb.py`)
- `notebooks/` – the analysis, written as percent scripts (`.py`) and saved as executed notebooks (`.ipynb`):
  - `01` score distributions and ceiling indicators
  - `02` measurement error at the top
  - `03` MACS case study
  - `04` retake simulation
  - `05` trends across years
  - `06` data and model checks
- `figures/` – the figures in the paper (PDF and PNG)
- `outputs/` – result tables (CSV) behind the numbers in the paper
- `data/raw/sources.csv` – the source files and where to download them

## Data
The data files are not included. All of them are public on naec.ge and edu.aris.ge and are listed in `data/raw/sources.csv`. The score lists identify students by exam ID, so they are not re-hosted here. To rerun the analysis, download the files into `data/raw/` using the filenames in `sources.csv`.

## Running
Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). Run from the repository root:
```
uv sync
uv run python src/extract_admitted_pdf.py
uv run python src/parse_statanalysis.py
uv run python src/build_nb.py notebooks/01_distributions.py notebooks/02_resolution.py notebooks/03_macs.py notebooks/04_stability.py notebooks/05_trends.py notebooks/06_model_checks.py
```

## Notes
The code was written with the help of an AI assistant. Data belong to NAEC. The code is released under the MIT License.
