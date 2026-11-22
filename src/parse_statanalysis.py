"""Parse NAEC 'statistical analysis' HTML reports (2022-2026) into a tidy CSV.

Two page templates exist:
- 2023-2026: <div id="{subject}" class="section level3"><h3>{subject}</h3>
  <h4>ტესტის მაქსიმალური ქულა NN</h4> <table>...
- 2022: subject in <h2> inside a heading-table, max score in
  <h3>ტესტის მაქსიმალური ქულა NN</h3>; stats table lacks the
  'ზღვარი გადალახა' column and has 'გამოცდაზე დაფიქსირებული მაქსიმალური ქულა'.

Stats-table columns (subset, by header prefix):
ვარიანტი | აბიტურიენტთა რაოდენობა | ზღვარი გადალახა "n (pct%)" |
საშუალო ქულა | საშუალო სირთულე | გამოცდაზე ... მაქსიმალური ქულა "score (count)" |
გამსვლელი ქულა (2025/2026 only)
Histogram tables (bins like "(55;70]") and heading tables are ignored.
2020/2021 reports are .ppsx slideshows, not this HTML format: skipped.
"""

import csv
import os
import re
import sys

from bs4 import BeautifulSoup

RAW = os.path.join("data", "raw")
OUT = os.path.join("data", "processed")

YEARS = [2022, 2023, 2024, 2025, 2026]

ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6,
         "VII": 7, "VIII": 8, "IX": 9, "X": 10,
         "XI": 11, "XII": 12}
ROMAN_RE = re.compile(r"^([IVX]+)$")
MAXSCORE_RE = re.compile(r"ტესტის მაქსიმალური ქულა\s*([\d.]+)")
CELL_RE = re.compile(r"^\s*([\d., ]+?)\s*(?:\(\s*[\d.,]+\s*%\s*\))?\s*$")
TOP_RE = re.compile(r"^\s*([\d.,]+)\s*\(\s*(\d+)\s*\)\s*$")


def is_subject_heading(text):
    return (0 < len(text) < 60 and "ტესტის" not in text
            and text != "სტატისტიკური ანალიზი")


def parse_num(s):
    s = s.strip().replace(",", ".")
    return float(s) if s else ""


def parse_file(year, path):
    soup = BeautifulSoup(open(path, encoding="utf-8").read(), "html.parser")
    rows = []
    warnings = []
    cur_subject = ""
    cur_max = ""
    for el in soup.find_all(["h2", "h3", "h4", "table"]):
        if el.name in ("h2", "h3"):
            txt = el.get_text(" ", strip=True)
            m = MAXSCORE_RE.search(txt)
            if m:
                cur_max = m.group(1)
            elif el.name == "h2" or is_subject_heading(txt):
                cur_subject = txt
        elif el.name == "h4":
            m = MAXSCORE_RE.search(el.get_text(" ", strip=True))
            if m:
                cur_max = m.group(1)
        else:  # table
            hdr = [th.get_text(strip=True) for th in el.find_all("th")]
            if not (any(h.startswith("ვარიანტი") for h in hdr)
                    and any(h.startswith("აბიტურიენტთა") for h in hdr)):
                continue

            def col_idx(prefix):
                for i, h in enumerate(hdr):
                    if h.startswith(prefix):
                        return i
                return None

            ci_n = col_idx("აბიტურიენტთა")
            # 2023 labels this column "კონკურსში მონაწილეობას აგრძელებს"
            ci_pass = col_idx("ზღვარი")
            if ci_pass is None:
                ci_pass = col_idx("კონკურსში")
            ci_mean = col_idx("საშუალო ქულა")
            ci_diff = col_idx("საშუალო სირთულე")
            ci_top = col_idx("გამოცდაზე")
            ci_pscore = col_idx("გამსვლელი")

            for tr in el.find_all("tr"):
                cells = [td.get_text(" ", strip=True)
                         for td in tr.find_all("td")]
                if len(cells) < 3 or not ROMAN_RE.match(cells[0]) \
                        or cells[0] not in ROMAN:
                    continue

                def cell(i):
                    return cells[i] if i is not None and i < len(cells) else ""

                try:
                    mpass = CELL_RE.match(cell(ci_pass))
                    mtop = TOP_RE.match(cell(ci_top))
                    if mtop:
                        top_obtained, n_top = parse_num(mtop.group(1)), int(mtop.group(2))
                    else:
                        top_obtained, n_top = parse_num(cell(ci_top)), ""
                    rows.append({
                        "year": year,
                        "subject_ka": cur_subject,
                        "max_score": cur_max,
                        "version": ROMAN[cells[0]],
                        "n": int(re.sub(r"[^\d]", "", cell(ci_n)) or 0),
                        "n_passed": int(parse_num(mpass.group(1))) if mpass and mpass.group(1) else "",
                        "mean": parse_num(cell(ci_mean)),
                        "mean_difficulty": parse_num(cell(ci_diff)),
                        "top_obtained": top_obtained,
                        "n_top_obtained": n_top,
                        "pass_score": parse_num(cell(ci_pscore)),
                    })
                except Exception as e:
                    warnings.append(f"{year} {cur_subject} {cells}: {e}")
    return rows, warnings


def main():
    os.makedirs(OUT, exist_ok=True)
    all_rows = []
    for year in YEARS:
        path = os.path.join(RAW, f"statanalysis_{year}.html")
        rows, warns = parse_file(year, path)
        all_rows.extend(rows)
        subs = []
        for r in rows:
            if r["subject_ka"] not in subs:
                subs.append(r["subject_ka"])
        print(f"[{year}] {len(rows)} rows, {len(subs)} subjects: {subs}")
        for w in warns[:10]:
            print("   WARN:", w)
    out = os.path.join(OUT, "statanalysis.csv")
    cols = ["year", "subject_ka", "max_score", "version", "n", "n_passed",
            "mean", "mean_difficulty", "top_obtained", "n_top_obtained",
            "pass_score"]
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
