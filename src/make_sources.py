import csv
import os
import urllib.parse

raw = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
rows = []


def A(f, u="", a="", n=""):
    rows.append([f, u, a, "2026-11-15", n])


for y in range(2020, 2027):
    A(f"Stats_{y}.xlsx", n="NAEC equating table (naec.ge); direct URL not recorded")
A("Admissions_2026.xlsx", n="from NAEC/edu.aris.ge; direct URL not recorded (admitted students ranked list 2026)")
A("მინიმალური-საგრანტო-ქულები_2025.xlsx", n="from NAEC/edu.aris.ge; direct URL not recorded; minimal grant score thresholds 2025 (not in request; kept)")
A("პროგრამების არჩევანი პრიორიტეტების მიხედვით 2026.xlsx", n="from NAEC/edu.aris.ge; direct URL not recorded; same file as choices_2026.xlsx (Georgian filename kept)")
A("ჩარიცხულთა-სია-სახელმწიფო-სასწავლო-გრანტის-მითითებით-2025.xlsx", n="from NAEC/edu.aris.ge; direct URL not recorded; same file as admitted_2025.xlsx (Georgian filename kept)")

A("rating_2024.xlsx",
  "https://edu.aris.ge/news/wp-content/uploads/2024/09/რეიტინგული-სია-საგნების-მიხედვით.xlsx",
  "https://edu.aris.ge/news/cnobilia-rogoria-2024-wlis-abiturientebis-reitinguli-sia-chasabarebeli-sagnebis-mixedvit-dokumenti.html",
  "")
A("rating_2026.xlsx",
  "https://edu.aris.ge/news/wp-content/uploads/2026/08/რეიტინგული-სია-საგნების-მიხედვით-2026.xlsx",
  "https://edu.aris.ge/news/cnobilia-2026-wlis-abiturientebis-reitinguli-sia-chasabarebeli-sagnebis-mixedvit-dokumenti.html",
  "")
A("choices_2024.xlsx",
  "https://edu.aris.ge/news/wp-content/uploads/2024/09/პროგრამების-არჩევანი-პრიორიტეტების-მიხედვით.xlsx",
  "https://edu.aris.ge/news/cnobilia-2024-wlis-abiturientebis-saganmanatleblo-programabis-archevani-prioritetebis-mixedvit.html",
  "")

A("rating_2025.xlsx",
  "https://edu.aris.ge/news/wp-content/uploads/2025/08/რეიტინგული-სია-საგნების-მიხედვით-1.xlsx",
  "https://edu.aris.ge/news/cnobilia-2025-wlis-abiturientebis-reitinguli-sia-chasabarebeli-sagnebis-mixedvit-dokumenti.html")
A("choices_2026.xlsx",
  "https://naec.ge/uploads/postData/2026/აბიტურიენტები - შედეგები/პროგრამების არჩევანი პრიორიტეტების მიხედვით 2026.xlsx",
  "https://edu.aris.ge/news/2026-abiturientta-saganmanatleblo-programebis-archevani-prioritetebis-mixedvit.html",
  "identical to Georgian-named file; from NAEC/edu.aris.ge")
A("applications_2020.xlsx",
  "https://edu.aris.ge/news/wp-content/uploads/2020/09/საგანმანათლებლო-პროგრამები-2020.xlsx",
  "https://edu.aris.ge/news/2020-wlis-ertiani-erovnuli-gamocdebis-ganacxadebis-raodenoba-saganmanatleblo-programebis-mixedvit.html",
  "applicant counts per program 2020")

adm = [
    ("admitted_2020_S0.pdf", "https://edu.aris.ge/news/wp-content/uploads/2020/09/Admissions_S_0.pdf", 2020, "ranked list part S0 (PDF)"),
    ("admitted_2020_S2.pdf", "https://edu.aris.ge/news/wp-content/uploads/2020/09/Admissions_S_2.pdf", 2020, "ranked list part S2 (PDF)"),
    ("admitted_2021_S0.pdf", "https://edu.aris.ge/news/uploads/postData/20-21/რანჟირებული სია 2021/Admissions_S_0.pdf", 2021, "ranked list part S0 (PDF)"),
    ("admitted_2021_S2.pdf", "https://edu.aris.ge/news/uploads/postData/20-21/რანჟირებული სია 2021/Admissions_S_2.pdf", 2021, "ranked list part S2 (PDF)"),
    ("admitted_2021_grants.pdf", "https://edu.aris.ge/news/uploads/postData/20-21/რანჟირებული სია 2021/Grants_web.pdf", 2021, "grants list (PDF)"),
    ("admitted_2022_ranking_doc.pdf", "https://edu.aris.ge/news/wp-content/uploads/2022/08/რანჟირების-დოკუმენტი.pdf", 2022, "ranking doc (PDF)"),
    ("admitted_2022_bachelor.pdf", "https://edu.aris.ge/news/wp-content/uploads/2022/08/რანჟირებული-სია-საბაკალავრო-პროგრამები.pdf", 2022, "ranked list bachelor programs (PDF)"),
    ("admitted_2022_ka_prep.pdf", "https://edu.aris.ge/news/wp-content/uploads/2022/08/რანჟირებული-სია-ქართულ-ენაში-მომზადების-პროგრამა.pdf", 2022, "ranked list Georgian-language prep (PDF)"),
    ("admitted_2023_grants.pdf", "https://edu.aris.ge/news/wp-content/uploads/2023/09/რანჟირების-დოკუმენტი-იმ-აბიტურიენტთა-ჩამონათვალის-მითითებით-რომლებმაც-მოიპოვეს-გრანტი.pdf", 2023, "ranking doc with grant winners (PDF)"),
    ("admitted_2023_bachelor.pdf", "https://edu.aris.ge/news/wp-content/uploads/2023/09/რანჟირებული-სია-აკადემიური-საბაკალავრო-პროგრამები.pdf", 2023, "ranked list academic bachelor (PDF)"),
    ("admitted_2023_ka_prep.pdf", "https://edu.aris.ge/news/wp-content/uploads/2023/09/რანჟირებული-სია-ქართულ-ენაში-მომზადების-საგანმანათლებლო-პროგრამები.pdf", 2023, "ranked list Georgian prep (PDF)"),
    ("admitted_2024_ranking_doc.pdf", "https://edu.aris.ge/news/wp-content/uploads/2024/08/აბსოლუტური-ქულებით-რანჟირების-დოკუმენტი-2.pdf", 2024, "ranking doc absolute scores (PDF)"),
    ("admitted_2024_bachelor.pdf", "https://edu.aris.ge/news/wp-content/uploads/2024/08/რანჟირებული-სია-აკადემიური-საბაკალავრო-დოკუმენტი.pdf", 2024, "ranked list academic bachelor (PDF)"),
    ("admitted_2024_ka_prep.pdf", "https://edu.aris.ge/news/wp-content/uploads/2024/08/რანჟირებული-სია-ქართულ-ენაში-მომზადების-საგანმანათლებლო-პროგრამები.pdf", 2024, "ranked list Georgian prep (PDF)"),
    ("admitted_2025.xlsx", "https://edu.aris.ge/news/wp-content/uploads/2025/08/ჩარიცხულთა-სია-სახელმწიფო-სასწავლო-გრანტის-მითითებით.xlsx", 2025, "admitted list with grant flag (xlsx)"),
]
for f, u, y, n in adm:
    A(f, u, f"https://edu.aris.ge/news/{y}-wels-charicxulta-ranjirebuli-sia-fakultetebisa-da-archeviti-sagnebis-mixedvit.html", n)

A("statanalysis_2020.ppsx", "https://edu.aris.ge/news/uploads/postData/20-20/სტატისტიკური ანალიზი_2020.ppsx",
  "https://edu.aris.ge/news/2020-wlis-ertiani-erovnuli-gamocdebis-statistikuri-analizi.html", "PowerPoint slideshow, not HTML")
A("statanalysis_2021.ppsx", "https://edu.aris.ge/news/wp-content/uploads/2021/08/statA-naec2021.ppsx",
  "https://edu.aris.ge/news/2021-wlis-ertiani-erovnuli-gamocdebis-statistikuri-analizi.html", "PowerPoint slideshow, not HTML")
A("statanalysis_2022.html", "https://naec.ge/uploads/postData/20-22/სტატისტიკური ანალიზი/exams_2022.html",
  "https://edu.aris.ge/news/2022-wlis-ertiani-erovnuli-gamocdebis-statistikuri-analizi.html")
A("statanalysis_2023.html", "https://naec.ge/uploads/postData/20-23/სტატისტიკური ანალიზი/exams_report_29.08.23 (2).html",
  "https://edu.aris.ge/news/2023-wlis-ertiani-erovnuli-gamocdebis-statistikuri-analizi.html")
A("statanalysis_2024.html", "https://naec.ge/uploads/postData/20-24/სტატისტიკური ანალიზი/exams_report_2024 (2).html",
  "https://edu.aris.ge/news/2024-wlis-ertiani-erovnuli-gamocdebis-statistikuri-analizi.html")
A("statanalysis_2025.html", "https://naec.ge/uploads/postData/20-25/სტატისტიკური ანალიზი - 2025.html",
  "https://edu.aris.ge/news/2025-wlis-ertiani-erovnuli-gamocdebis-statistikuri-analizi.html")
A("statanalysis_2026.html", "https://naec.ge/uploads/postData/2026/აბიტურიენტები - შედეგები/Final_Report.html",
  "https://edu.aris.ge/news/2026-wlis-ertiani-erovnuli-gamocdebis-statistikuri-analizi.html")

mt_dir = os.path.join(raw, "math_tests")
cur_year = None
seen = set()
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ce_math_urls.txt"), encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line.startswith("== "):
            cur_year = line[3:]
            continue
        if not line.startswith("http"):
            continue
        url = line
        leaf = urllib.parse.unquote(url).split("/")[-1]
        if "სქემა" in leaf:
            if "დამატებით" in url:
                name = "scheme_extra"
            elif "III" in leaf or "ვარიანტი 3" in leaf:
                name = "scheme_v3"
            elif "II" in leaf:
                name = "scheme_v2"
            else:
                name = "scheme_v1"
        else:
            if "დამატებით" in url:
                name = "test_extra"
            elif leaf.startswith("III"):
                name = "test_v3"
            elif leaf.startswith("II"):
                name = "test_v2"
            elif leaf.startswith("I ") or leaf == "I.pdf":
                name = "test_v1"
            elif "_2" in leaf or leaf.endswith(" 2.pdf"):
                name = "test_v2"
            else:
                name = "test_v1"
        fn = f"math_tests/math_{cur_year}_{name}.pdf"
        if fn in seen:
            continue
        seen.add(fn)
        exists = os.path.exists(os.path.join(mt_dir, f"math_{cur_year}_{name}.pdf"))
        A(fn, urllib.parse.unquote(url),
          f"https://edu.aris.ge/news/{cur_year}-wlis-ertian-erovnul-gamocdebze-gamoyenebuli-testebi.html",
          "" if exists else "LINKED in article but returns 404 - NOT downloaded")

with open(os.path.join(raw, "sources.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "url", "article_url", "downloaded_at", "notes"])
    w.writerows(rows)
print(len(rows), "rows written")
