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

## Paper figures and tables

| Paper element | Notebook | Output file(s) |
|---|---|---|
| Fig. 1 `fig01_math_upper_tail` | `notebooks/01_distributions.py` | – |
| Fig. 2 `fig02_ceiling_indicators` | `notebooks/01_distributions.py` | `outputs/t01_subject_year.csv` (data) |
| Fig. 3 `fig03_measurement_error` | `notebooks/02_resolution.py` | `outputs/t02_math_sem.csv`, `outputs/t02_math_equating_bootstrap_2026.csv` |
| Fig. 4 `fig04_macs_crowding` | `notebooks/03_macs.py` | `outputs/t03_macs_cutoff_crowd.csv` (cutoff values) |
| Fig. 5 `fig05_macs_admits_2026` | `notebooks/03_macs.py` | – |
| Fig. 6 `fig06_stability` | `notebooks/04_stability.py` | `outputs/t04_stability_summary.csv`, `outputs/t04_stability_by_distance.csv` |
| Fig. 7 `fig07_trends` | `notebooks/05_trends.py` | `outputs/t05_macs_by_year.csv` |
| Fig. 8 `fig08_programs_stability` | `notebooks/04_stability.py` | `outputs/t04_stability_summary.csv` |
| Table 1 (test format & MACS weights) | – | written by hand; format facts in `outputs/t02_math_test_structure.csv` (nb02), weights in `outputs/t05_macs_by_year.csv` (nb05) |
| Table 2 (data sources) | – | `data/raw/sources.csv` |
| Table 3 (top of math distribution) | `notebooks/01_distributions.py` | `outputs/t01_math_top_counts.csv` |
| Table 4 (MACS admissions, math route) | `notebooks/05_trends.py` | `outputs/t05_macs_by_year.csv` |
| Table 5 (MACS places changing hands) | `notebooks/04_stability.py` | `outputs/t04_stability_summary.csv` |
| Table 6 (places changing hands, by program) | `notebooks/04_stability.py` | `outputs/t04_stability_summary.csv` |
| Table 7 (data and model checks, appendix) | `notebooks/06_model_checks.py` | `outputs/t06_counts_check.csv`, `outputs/t06_raw_recovery_check.csv`, `outputs/t06_model_reliability.csv`, `outputs/t06_distribution_preservation.csv` |

The remaining `outputs/t0*.csv` files back in-text numbers in the Results and appendix.

## Notes
The code was written with the help of an AI assistant. Data belong to NAEC. The code is released under the MIT License.
