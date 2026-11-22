"""Extract admitted-student rows for university 036 from the NAEC ranked-list PDFs (2020-2024).

Layout (all five years, verified): header line "საგამოცდოს ქართული უცხოური {elective...} საკონკურსო გრანტი %"
followed by "ნომერი ენა [elective-name tail] ქულა", then rows: rank, 9-digit exam id, geo scaled,
foreign-lang scaled, elective scaled score(s), competitive score, optional grant %.
Long elective names wrap: "სამოქალაქო" stays on the header line and "განათლება" lands on the
"ნომერი ენა ... ქულა" line; the tail is re-attached to the last elective.
ვიზუალური ხელოვნება და დიზაინი has no elective column at all (rows are geo fl comp [grant]);
the remaining comp component (practical/creative exam) is not shown.
Fields not present in these PDFs (blank in output): geo_raw, fl (language not named),
fl_raw, *_raw, choice, type. The last PDF column (grant %) is kept as an extra `grant` field.

Second target: university 001 (TSU), program name containing "ISET" (one code per year:
00101284/00101294/00101404/00101404/00101384). Written to admitted_iset_all.csv.
In 2024 its name wraps, so "(ინგლისურენოვანი)" is on the line after the header and is
not part of the stored prog value.
"""

import csv
import os
import re
import sys

import pdfplumber

RAW = os.path.join("data", "raw")
OUT = os.path.join("data", "processed")
UNI_WANT = "036"

FILES = {
    2020: "admitted_2020_S0.pdf",
    2021: "admitted_2021_S0.pdf",
    2022: "admitted_2022_bachelor.pdf",
    2023: "admitted_2023_bachelor.pdf",
    2024: "admitted_2024_bachelor.pdf",
}

COLS = ["year", "uni_code", "uni", "prog_code", "prog", "exam_id",
        "geo_raw", "geo_sc", "fl", "fl_raw", "fl_sc",
        "el1", "el1_raw", "el1_sc", "el2", "el2_raw", "el2_sc",
        "comp", "choice", "type", "grant"]

RE_UNI = re.compile(r"^(\d{3})\s+(.*\D.*)$")
RE_PROG = re.compile(r"^(\d{8,9})\s+(.*)$")
RE_ROW = re.compile(r"^(\d+)\s+(\d{9})\s*(.*)$")
RE_COLHDR = re.compile(r"საგამოცდოს")
SKIP = ("აბიტურენტთა", "აკადემიური", "ნომერი ენა ქულა", "წლიური გადასახადი",
        "გამოცდების ეროვნული ცენტრი", "1 2 3 4")


def target_tag(uni_code, prog):
    if uni_code == UNI_WANT:
        return "036"
    if uni_code == "001" and prog and "ISET" in prog:
        return "iset"
    return None


def parse_year(year, path):
    rows = []
    bad_rows = []
    uni_code = uni = prog_code = prog = None
    electives = []
    in_rows = False
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                m_uni = RE_UNI.match(line)
                m_prog = RE_PROG.match(line)
                if m_uni and not m_prog:
                    uni_code, uni = m_uni.group(1), m_uni.group(2).strip()
                    prog_code = prog = None
                    electives = []
                    in_rows = False
                    continue
                if m_prog:
                    prog_code, prog = m_prog.group(1), m_prog.group(2).strip()
                    electives = []
                    in_rows = False
                    continue
                if RE_COLHDR.search(line):
                    # electives = tokens between უცხოური and საკონკურსო
                    mm = re.search(r"უცხოური\s+(.*?)\s+საკონკურსო", line)
                    electives = mm.group(1).split() if mm else []
                    in_rows = True
                    continue
                m_cont = re.match(r"^ნომერი\s+ენა\s+(.*?)\s*ქულა\s*$", line)
                if m_cont:
                    extra = m_cont.group(1).strip()
                    if extra and electives:
                        electives[-1] += " " + extra
                    continue
                if any(s in line for s in SKIP):
                    continue
                if re.match(r"^\d+\s*/\s*\d+$", line):  # page footer "155 / 720"
                    continue
                tag = target_tag(uni_code, prog) \
                    if prog_code is not None and in_rows else None
                if tag is None:
                    continue
                m_row = RE_ROW.match(line)
                if not m_row:
                    continue
                toks = m_row.group(3).split()
                nums, ok = [], True
                for t in toks:
                    try:
                        nums.append(float(t))
                    except ValueError:
                        ok = False
                n_el = len(electives)
                # expected: geo, fl, el*n_el, comp, [grant]
                if ok and len(nums) in (3 + n_el, 4 + n_el):
                    geo, fl = nums[0], nums[1]
                    els = nums[2:2 + n_el]
                    comp = nums[2 + n_el]
                    grant = int(nums[3 + n_el]) if len(nums) == 4 + n_el else ""
                else:
                    bad_rows.append(line)
                    if len(nums) >= 4:
                        geo, fl = nums[0], nums[1]
                        els = nums[2:2 + n_el] if n_el else nums[2:-1]
                        comp = nums[2 + n_el] if len(nums) > 2 + n_el else ""
                        grant = ""
                    else:
                        continue
                el1 = electives[0] if len(electives) > 0 else ""
                el2 = electives[1] if len(electives) > 1 else ""
                rows.append({
                    "target": tag,
                    "year": year, "uni_code": uni_code, "uni": uni,
                    "prog_code": prog_code, "prog": prog,
                    "exam_id": m_row.group(2),
                    "geo_raw": "", "geo_sc": geo, "fl": "", "fl_raw": "", "fl_sc": fl,
                    "el1": el1, "el1_raw": "", "el1_sc": els[0] if len(els) > 0 else "",
                    "el2": el2, "el2_raw": "", "el2_sc": els[1] if len(els) > 1 else "",
                    "comp": comp, "choice": "", "type": "",
                    "grant": grant,
                })
    return rows, bad_rows


def write_csv(path, rows):
    rows = [{k: v for k, v in r.items() if k in COLS} for r in rows]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    all_rows = []
    for year, fname in FILES.items():
        rows, bad = parse_year(year, os.path.join(RAW, fname))
        all_rows.extend(rows)
        n036 = sum(1 for r in rows if r["target"] == "036")
        niset = sum(1 for r in rows if r["target"] == "iset")
        print(f"[{year}] {fname}: {n036} rows for uni 036, "
              f"{niset} rows for ISET, {len(bad)} unparseable row lines")
        for b in bad[:10]:
            print("    BAD:", b[:110])
    rows036 = [r for r in all_rows if r["target"] == "036"]
    out = os.path.join(OUT, "admitted_036_all.csv")
    write_csv(out, rows036)
    # also per-year files
    for year in FILES:
        write_csv(os.path.join(OUT, f"admitted_{year}_036.csv"),
                  [r for r in rows036 if r["year"] == year])
    iset = os.path.join(OUT, "admitted_iset_all.csv")
    write_csv(iset, [r for r in all_rows if r["target"] == "iset"])
    print("wrote", out, "and", iset)


if __name__ == "__main__":
    sys.exit(main())
