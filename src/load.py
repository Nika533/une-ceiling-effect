"""Loaders for NAEC (Georgia) Unified National Exam data in data/raw.

All loaders return tidy pandas DataFrames with English column names.
Subject codes: GEO (Georgian language & literature), ENG, RUS, GER, FRE, MATH, HIST,
BIO, CHEM, PHYS, GEOG, CIV, LIT, ART, GAT_AZ, GAT_AM, OSS, S98 (unlabelled code '98' in 2024).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

SHEET_SUBJECT = {"GEO": "GEO", "ENG": "ENG", "RUS": "RUS", "HIST": "HIST", "MATH": "MATH",
                 "GAT": "GAT_AZ", "GAT AZER": "GAT_AZ"}

# (substring, code); first match wins, so order matters.
_SUBJECT_PATTERNS = [
    ("ქართული ენა", "GEO"), ("ინგლის", "ENG"), ("რუს", "RUS"), ("გერმან", "GER"),
    ("ფრანგ", "FRE"), ("მათემ", "MATH"), ("ისტორ", "HIST"), ("ბიოლ", "BIO"), ("ქიმ", "CHEM"),
    ("ფიზ", "PHYS"), ("გეოგრ", "GEOG"), ("სამოქ", "CIV"), ("ლიტერ", "LIT"), ("ხელოვ", "ART"),
    ("აზერ", "GAT_AZ"), ("სომხ", "GAT_AM"), ("ოსურ", "OSS"), ("ზოგადი უნარები", "GAT"),
]


def subject_code(name) -> str:
    s = str(name).strip()
    if re.fullmatch(r"\d+", s):
        return f"S{s}"
    for pat, code in _SUBJECT_PATTERNS:
        if pat in s:
            return code
    raise ValueError(f"unknown subject name: {s!r}")


def _parse_equating_sheet(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all", axis=1).dropna(how="all", axis=0).reset_index(drop=True)
    df.columns = range(df.shape[1])
    sub_row = next(i for i in range(len(df)) if df.iloc[i].astype(str).str.contains("სიმკვრივე").any())
    sub = [str(x) for x in df.iloc[sub_row].tolist()]
    body = df.iloc[sub_row + 1:]
    body = body[pd.to_numeric(body[0], errors="coerce").notna()]
    raw = pd.to_numeric(body[0]).astype(int)
    out = []
    starts = [j for j, s in enumerate(sub) if "სიმკვრივე" in s]
    for v, j in enumerate(starts, start=1):
        assert "რანგ" in sub[j + 1] and "გათანაბრ" in sub[j + 2], sub
        part = pd.DataFrame({
            "version": v,
            "raw": raw.values,
            "n": pd.to_numeric(body[j], errors="coerce").values,
            "pct_rank": pd.to_numeric(body[j + 1], errors="coerce").values,
            "equated": pd.to_numeric(body[j + 2], errors="coerce").values,
        })
        out.append(part.dropna(subset=["n"]))
    res = pd.concat(out, ignore_index=True)
    res["n"] = res["n"].astype(int)
    return res


def load_equating(years=range(2020, 2027)) -> pd.DataFrame:
    """Equating tables: one row per (year, subject, version, raw score)."""
    frames = []
    for year in years:
        f = RAW / f"Stats_{year}.xlsx"
        for sheet, df in pd.read_excel(f, sheet_name=None, header=None).items():
            part = _parse_equating_sheet(df)
            part.insert(0, "subject", SHEET_SUBJECT[sheet.strip()])
            part.insert(0, "year", year)
            frames.append(part)
    eq = pd.concat(frames, ignore_index=True)
    eq["max_raw"] = eq.groupby(["year", "subject"])["raw"].transform("max")
    return eq


def load_rating(year: int) -> pd.DataFrame:
    """Full-population rating list: one row per (student, subject) with the scaled score."""
    r = pd.read_excel(RAW / f"rating_{year}.xlsx", dtype=str)
    assert r.shape[1] == 4, r.columns
    r.columns = ["rank", "subject_ka", "exam_id", "scaled"]
    r = r.dropna(subset=["exam_id"])
    r["subject"] = r["subject_ka"].map(subject_code)
    r["rank"] = pd.to_numeric(r["rank"])
    r["scaled"] = pd.to_numeric(r["scaled"])
    r["exam_id"] = r["exam_id"].str.strip()
    r.insert(0, "year", year)
    return r[["year", "exam_id", "subject", "scaled", "rank", "subject_ka"]]


def rating_wide(year: int) -> pd.DataFrame:
    """Rating list pivoted to one row per student, one column per subject (scaled score)."""
    r = load_rating(year)
    dup = r.duplicated(["exam_id", "subject"]).sum()
    assert dup == 0, f"{dup} duplicate (student, subject) rows in rating_{year}"
    return r.pivot(index="exam_id", columns="subject", values="scaled")


ADMISSION_COLS = ["uni_code", "uni", "prog_code", "prog", "exam_id", "geo_raw", "geo_sc",
                  "fl", "fl_raw", "fl_sc", "el1", "el1_raw", "el1_sc", "el2", "el2_raw",
                  "el2_sc", "comp", "grant", "choice", "type"]
_NUM = ["geo_raw", "geo_sc", "fl_raw", "fl_sc", "el1_raw", "el1_sc", "el2_raw", "el2_sc", "comp", "grant", "choice"]


def load_admissions(year: int) -> pd.DataFrame:
    """Admitted students (one row per admitted student)."""
    f = RAW / ("Admissions_2026.xlsx" if year == 2026 else f"admitted_{year}.xlsx")
    a = pd.read_excel(f, dtype=str)
    if a.shape[1] != len(ADMISSION_COLS):
        raise ValueError(f"{f.name}: unexpected columns {list(a.columns)}")
    a.columns = ADMISSION_COLS
    for c in _NUM:
        a[c] = pd.to_numeric(a[c], errors="coerce")
    for c in ["uni_code", "prog_code", "exam_id"]:
        a[c] = a[c].str.strip()
    for c in ["fl", "el1", "el2"]:
        a[c + "_code"] = a[c].map(lambda s: subject_code(s) if isinstance(s, str) else np.nan)
    a.insert(0, "year", year)
    return a


def load_choices(year: int) -> pd.DataFrame:
    """Number of applicants listing each program at priority I..X."""
    c = pd.read_excel(RAW / f"choices_{year}.xlsx", dtype=str)
    assert c.shape[1] == 15, c.columns
    c.columns = ["uni_code", "uni", "prog_code", "prog"] + [f"p{i}" for i in range(1, 11)] + ["total"]
    for col in [f"p{i}" for i in range(1, 11)] + ["total"]:
        c[col] = pd.to_numeric(c[col])
    for col in ["uni_code", "prog_code"]:
        c[col] = c[col].str.strip()
    c.insert(0, "year", year)
    return c
