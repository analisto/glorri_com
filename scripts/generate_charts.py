"""
Glorri.com Job Market Analysis — Chart Generator
Produces business-oriented charts saved to charts/
"""

import os
import csv
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH = "data/glorri.csv"
OUT_DIR = "charts"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
PRIMARY   = "#1A56DB"
SECONDARY = "#16A34A"
ACCENT    = "#EA580C"
MUTED     = "#94A3B8"
BG        = "#FFFFFF"
GRID      = "#E2E8F0"
TEXT      = "#1E293B"

FONT = {"family": "DejaVu Sans"}
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "axes.facecolor": BG,
    "figure.facecolor": BG,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})

# ── Translation maps ─────────────────────────────────────────────────────────
CAREER_MAP = {
    "": "Unspecified",
    "career-levels.undefined": "Unspecified",
    "Mütəxəssis": "Specialist",
    "Gənc mütəxəssis": "Junior Specialist",
    "Orta səviyyəli": "Mid-Level",
    "Rəhbər": "Executive / Head",
    "Supervayzer": "Supervisor",
    "Məzun": "Graduate",
    "Direktor": "Director",
    "Tələbə": "Student / Intern",
    "Tərəfdaş": "Partner",
    "Digər": "Other",
}

TYPE_MAP = {
    "Tam ştat": "Full-time",
    "Yarım ştat": "Part-time",
    "Təcrübə": "Internship",
    "Mövsümi": "Seasonal",
    "Növbə əsasında": "Shift-based",
    "Müvəqqəti": "Temporary",
    "Daimi": "Permanent",
    "Təqaüd proqramı": "Traineeship",
    "Könüllü": "Volunteer",
    "": "Unspecified",
}

FUNC_MAP = {
    "Bank xidmətləri": "Banking",
    "Digər xidmətlər": "Other Services",
    "İnformasiya Texnologiyaları": "Information Technology",
    "İnşaat/tikinti, daşınmaz əmlak": "Construction & Real Estate",
    "Satışlar": "Sales",
    "Təhsil, kurslar, təlim- tədris": "Education & Training",
    "Mühasibatlıq, müəssisənin iqtisadiyyatı": "Accounting & Economics",
    "Kənd təsərrüfatı və qida məhsulları": "Agriculture & Food",
    "İstehsal və sənaye məhsulları": "Manufacturing & Industry",
    "Səhiyyə, əczaçılıq": "Healthcare & Pharmacy",
    "Proqramlaşdırma, mobil tətbiqlər və veb dizayn": "Software & Web Dev",
    "Layihələrin idarə olunması": "Project Management",
    "Anbar xidmətləri": "Warehouse & Logistics",
    "Satınalma, təchizat, tədarük şəbəkəsi": "Procurement & Supply Chain",
    "Səyahət, mehmanxanalar, istirahət": "Travel, Hotels & Leisure",
    "İnsan resursları, kadr təchizatı, kadrların təlimi": "Human Resources",
    "Data analizi, statistik təhlil": "Data Analysis",
    "İnzibati heyət, ofis işləri": "Administration & Office",
    "Maliyyə xidmətləri": "Financial Services",
    "Müştərilərlə iş": "Customer Service",
    "Marketinq": "Marketing",
    "Hüquq, qanunvericilik": "Legal",
    "Sürücülük, daşıma": "Transport & Driving",
    "Logistika, xarici ticarət": "Logistics & Trade",
    "İctimai iaşə, aşpazlıq": "Hospitality & Catering",
    "Mühəndislik": "Engineering",
    "Texniki heyət": "Technical Staff",
    "Sosial media, SMM": "Social Media / SMM",
    "Dizayn": "Design",
    "Reklam, PR": "Advertising & PR",
    "Ticarət nümayəndəsi": "Trade Representative",
    "Sığorta": "Insurance",
    "Ekologiya, ətraf mühit": "Ecology & Environment",
    "Müdafiə, hərbi": "Defense & Military",
    "Əczaçı": "Pharmacist",
    "Keyfiyyətin idarəedilməsi": "Quality Management",
    "Texniki dəstək": "Technical Support",
    "Audit": "Audit",
    "Restoran, restoran işçiləri, iaşə xidmətləri": "Restaurant & Food Service",
    "Təmizlik xidmətləri": "Cleaning Services",
    "İdman, fitnes": "Sports & Fitness",
    "Mülkiyyət idarəetməsi": "Property Management",
    "Tikinti mühəndisliyi": "Civil Engineering",
    "Foto/video": "Photo / Video",
    "Jurnalistika, media": "Journalism & Media",
    "Arxitektura": "Architecture",
    "Neft-qaz sektoru": "Oil & Gas",
    "Aviasiya": "Aviation",
    "Dəniz nəqliyyatı": "Maritime",
    "Səhiyyə assistansiyası": "Healthcare Assistance",
    "Psixologiya": "Psychology",
    "Kənd təsərrüfatı": "Agriculture",
}

LOC_MAP = {
    "Bakı, Azərbaycan": "Baku",
    "Bakı, Azərbaycanan": "Baku",
    "Sumqayıt, Azərbaycan": "Sumgait",
    "Gəncə, Azərbaycan": "Ganja",
    "Yevlax, Azərbaycan": "Yevlax",
    "Xaçmaz, Azərbaycan": "Khachmaz",
    "Xırdalan, Azərbaycan": "Khirdalan",
    "Xankəndi, Azərbaycan": "Khankendi",
    "Balakən, Azərbaycan": "Balakan",
    "Bərdə, Azərbaycan": "Barda",
    "Beyləqan, Azərbaycan": "Beylagan",
    "Ağdaş, Azərbaycan": "Aghdash",
    "Ağdam, Azərbaycan": "Aghdam",
    "Şamaxı, Azərbaycan": "Shamakhi",
    "Lənkəran, Azərbaycan": "Lankaran",
    "Binəqədi, Bakı, Azərbaycan": "Binagadi (Baku)",
    "Lökbatan, Azərbaycan": "Lokbatan",
    "Mingəçevir, Azərbaycan": "Mingachevir",
    "Şirvan, Azərbaycan": "Shirvan",
    "Quba, Azərbaycan": "Quba",
    "Füzuli, Azərbaycan": "Fuzuli",
    "Ağcabədi, Azərbaycan": "Aghcabadi",
    "Şəki, Azərbaycan": "Shaki",
    "Goranboy, Azərbaycan": "Goranboy",
    "Naxçıvan, Azərbaycan": "Nakhchivan",
    "Zaqatala, Azərbaycan": "Zagatala",
    "Qazax, Azərbaycan": "Gazakh",
    "Saatlı, Azərbaycan": "Saatli",
    "Sabirabad, Azərbaycan": "Sabirabad",
    "Salyan, Azərbaycan": "Salyan",
    "Şəmkir, Azərbaycan": "Shamkir",
    "Cəlilabad, Azərbaycan": "Jalilabad",
    "Abşeron, Azərbaycan": "Absheron",
    "Lerik, Azərbaycan": "Lerik",
    "Masallı, Azərbaycan": "Masalli",
    "Neftçala, Azərbaycan": "Neftchala",
    "İmişli, Azərbaycan": "Imishli",
    "Qobustan, Azərbaycan": "Gobustan",
    "Kürdəmir, Azərbaycan": "Kurdamir",
    "Ağsu, Azərbaycan": "Aghsu",
    "Astara, Azərbaycan": "Astara",
    "Biləsuvar, Azərbaycan": "Bilasuvar",
    "Hacıqabul, Azərbaycan": "Hajigabul",
    "Şuşa, Azərbaycan": "Shusha",
    "Xocavənd, Azərbaycan": "Khojavend",
    "Kəlbəcər, Azərbaycan": "Kalbajar",
    "Laçın, Azərbaycan": "Lachin",
    "Qubadlı, Azərbaycan": "Qubadli",
    "Zəngilan, Azərbaycan": "Zangilan",
    "Ağstafa, Azərbaycan": "Aghstafa",
    "Tovuz, Azərbaycan": "Tovuz",
    "Gədəbəy, Azərbaycan": "Gadabay",
    "Daşkəsən, Azərbaycan": "Dashkasan",
    "": "Unspecified",
}

COMPANY_MAP = {
    "Mətanət A Şirkətlər Qrupu": "Matanat A Group",
    "Unibank Kommersiya Bankı": "Unibank",
    "Kontakt Home": "Kontakt Home",
    "Landau Education Group": "Landau Education Group",
    "AccessBank ": "AccessBank",
    "Xalq Bank": "Xalq Bank",
    "Prime Cotton": "Prime Cotton",
    "Avromed Company MMC": "Avromed",
    "İnnovasiya və Rəqəmsal İnkişaf Agentliyi": "Digital Innovation Agency",
    "Expressbank ASC": "Expressbank",
    "Azərbaycan Dəmir Yolları": "Azerbaijan Railways",
    "Agrotrade Supply MMC": "Agrotrade Supply",
    "PASHA Real Estate Group": "PASHA Real Estate",
    "OBA Market MMC": "OBA Market",
    "PASHA Property Management": "PASHA Property Mgmt",
    "Kapital Bank": "Kapital Bank",
    "Azərenerji": "Azerenerji",
    "Silk Way Airlines": "Silk Way Airlines",
    "SOCAR": "SOCAR",
    "Azərsu": "Azersu",
}

# ── Load data ────────────────────────────────────────────────────────────────
def load_data():
    rows = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["_views"] = int(row["viewCount"]) if row["viewCount"].isdigit() else 0
            row["_func_en"] = FUNC_MAP.get(row["jobFunction"], row["jobFunction"])
            row["_level_en"] = CAREER_MAP.get(row["careerLevel"], row["careerLevel"])
            row["_type_en"] = TYPE_MAP.get(row["type"], row["type"])
            raw_loc = row["location"].strip()
            row["_loc_en"] = LOC_MAP.get(raw_loc, raw_loc.split(",")[0])
            raw_co = row["company_name"].strip()
            row["_co_en"] = COMPANY_MAP.get(raw_co, raw_co)
            try:
                row["_date"] = datetime.fromisoformat(row["postedDate"].replace("Z", "+00:00"))
            except Exception:
                row["_date"] = None
            rows.append(row)
    return rows

# ── Helpers ──────────────────────────────────────────────────────────────────
def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


def hbar(labels, values, title, xlabel, color=PRIMARY, figsize=(12, 7), annotate=True):
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(labels, values, color=color, height=0.6)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(GRID)
    if annotate:
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:,}", va="center", fontsize=9, color=TEXT)
    ax.set_xlim(0, max(values) * 1.14)
    fig.tight_layout()
    return fig


def vbar(labels, values, title, ylabel, color=PRIMARY, figsize=(12, 6), annotate=True, rotation=30):
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=color, width=0.6)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotation, ha="right", fontsize=9)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(GRID)
    if annotate:
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                    f"{val:,}", ha="center", va="bottom", fontsize=9, color=TEXT)
    ax.set_ylim(0, max(values) * 1.14)
    fig.tight_layout()
    return fig


# ── Chart 1: Top 15 Companies by Job Postings ────────────────────────────────
def chart_top_companies(rows):
    counts = Counter(r["_co_en"] for r in rows)
    top = counts.most_common(15)
    labels = [t[0] for t in reversed(top)]
    values = [t[1] for t in reversed(top)]
    fig = hbar(labels, values,
               "Top 15 Most Active Hiring Companies",
               "Number of Job Postings",
               color=PRIMARY, figsize=(13, 8))
    save(fig, "01_top_companies_by_postings.png")


# ── Chart 2: Openings by Industry Sector ─────────────────────────────────────
def chart_by_industry(rows):
    counts = Counter(r["_func_en"] for r in rows if r["_func_en"])
    top = counts.most_common(15)
    labels = [t[0] for t in reversed(top)]
    values = [t[1] for t in reversed(top)]
    palette = [PRIMARY if v > 50 else SECONDARY if v > 25 else MUTED for v in reversed(values)]
    fig = hbar(labels, values,
               "Job Openings by Industry Sector  (Top 15)",
               "Number of Openings",
               color=palette, figsize=(13, 8))
    save(fig, "02_openings_by_industry.png")


# ── Chart 3: Monthly Hiring Trend ────────────────────────────────────────────
def chart_monthly_trend(rows):
    monthly = defaultdict(int)
    for r in rows:
        if r["_date"]:
            key = r["_date"].strftime("%Y-%m")
            monthly[key] += 1

    keys = sorted(monthly)
    # filter to last 18 months for clarity
    cutoff = (datetime.utcnow() - timedelta(days=18*30)).strftime("%Y-%m")
    keys = [k for k in keys if k >= cutoff]
    values = [monthly[k] for k in keys]
    dates = [datetime.strptime(k, "%Y-%m") for k in keys]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, values, color=PRIMARY, linewidth=2.5, marker="o", markersize=5)
    ax.fill_between(dates, values, alpha=0.12, color=PRIMARY)
    ax.set_title("Monthly Job Posting Volume  (Last 18 Months)", fontsize=14, fontweight="bold", pad=14)
    ax.set_ylabel("Postings", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(GRID)
    for d, v in zip(dates, values):
        ax.text(d, v + 1, str(v), ha="center", fontsize=7.5, color=TEXT)
    fig.tight_layout()
    save(fig, "03_monthly_hiring_trend.png")


# ── Chart 4: Career Level Demand ─────────────────────────────────────────────
def chart_career_levels(rows):
    counts = Counter(r["_level_en"] for r in rows if r["_level_en"] not in ("Unspecified",))
    labels = [k for k, _ in counts.most_common()]
    values = [counts[k] for k in labels]
    fig = vbar(labels, values,
               "Demand by Seniority Level  (Excluding Unspecified)",
               "Number of Openings",
               color=PRIMARY, figsize=(12, 6), rotation=25)
    save(fig, "04_career_level_demand.png")


# ── Chart 5: Employment Type Breakdown ───────────────────────────────────────
def chart_employment_types(rows):
    counts = Counter(r["_type_en"] for r in rows if r["_type_en"] not in ("Unspecified", ""))
    labels = [k for k, _ in counts.most_common()]
    values = [counts[k] for k in labels]
    colors = [PRIMARY if i == 0 else SECONDARY if i == 1 else MUTED for i in range(len(labels))]
    fig = vbar(labels, values,
               "Job Postings by Employment Type",
               "Number of Openings",
               color=colors, figsize=(12, 6), rotation=20)
    save(fig, "05_employment_type_breakdown.png")


# ── Chart 6: Top 15 Companies by Total Candidate Views ───────────────────────
def chart_company_views(rows):
    co_views = defaultdict(int)
    for r in rows:
        co_views[r["_co_en"]] += r["_views"]
    top = sorted(co_views.items(), key=lambda x: x[1], reverse=True)[:15]
    labels = [t[0] for t in reversed(top)]
    values = [t[1] for t in reversed(top)]
    fig = hbar(labels, values,
               "Top 15 Companies by Total Candidate Views",
               "Total Views",
               color=SECONDARY, figsize=(13, 8))
    save(fig, "06_top_companies_by_views.png")


# ── Chart 7: Avg Views per Posting by Industry ───────────────────────────────
def chart_avg_views_industry(rows):
    func_views = defaultdict(list)
    for r in rows:
        if r["_func_en"]:
            func_views[r["_func_en"]].append(r["_views"])
    avg = {k: sum(v) / len(v) for k, v in func_views.items() if len(v) >= 5}
    top = sorted(avg.items(), key=lambda x: x[1], reverse=True)[:15]
    labels = [t[0] for t in reversed(top)]
    values = [round(t[1]) for t in reversed(top)]
    fig = hbar(labels, values,
               "Average Candidate Views per Job Posting by Industry",
               "Avg. Views per Posting",
               color=ACCENT, figsize=(13, 8))
    save(fig, "07_avg_views_per_job_by_industry.png")


# ── Chart 8: Geographic Distribution (Top 10 Cities) ─────────────────────────
def chart_geography(rows):
    counts = Counter(r["_loc_en"] for r in rows if r["_loc_en"])
    top = counts.most_common(10)
    labels = [t[0] for t in reversed(top)]
    values = [t[1] for t in reversed(top)]
    fig = hbar(labels, values,
               "Geographic Distribution of Job Openings  (Top 10 Cities)",
               "Number of Openings",
               color=PRIMARY, figsize=(12, 7))
    save(fig, "08_geographic_distribution.png")


# ── Chart 9: Industry Mix by Employment Type (Stacked) ───────────────────────
def chart_industry_employment_stack(rows):
    # Top 10 industries, stacked: Full-time vs Other
    func_counts = Counter(r["_func_en"] for r in rows if r["_func_en"])
    top10 = [k for k, _ in func_counts.most_common(10)]

    ft_counts = defaultdict(int)
    other_counts = defaultdict(int)
    for r in rows:
        fn = r["_func_en"]
        if fn in top10:
            if r["_type_en"] == "Full-time":
                ft_counts[fn] += 1
            else:
                other_counts[fn] += 1

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(top10))
    ft_vals = [ft_counts[fn] for fn in top10]
    ot_vals = [other_counts[fn] for fn in top10]

    b1 = ax.bar(x, ft_vals, label="Full-time", color=PRIMARY, width=0.6)
    b2 = ax.bar(x, ot_vals, bottom=ft_vals, label="Other Types", color=ACCENT, width=0.6)

    ax.set_title("Employment Type Mix Across Top 10 Industries", fontsize=14, fontweight="bold", pad=14)
    ax.set_ylabel("Number of Openings", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(top10, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(GRID)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    save(fig, "09_industry_employment_type_stack.png")


# ── Chart 10: Weekly Posting Volume (Last 12 Weeks) ──────────────────────────
def chart_weekly_trend(rows):
    cutoff = datetime.utcnow() - timedelta(weeks=12)
    weekly = defaultdict(int)
    for r in rows:
        if r["_date"] and r["_date"].replace(tzinfo=None) >= cutoff:
            # ISO week
            key = r["_date"].strftime("%Y-W%W")
            weekly[key] += 1

    keys = sorted(weekly)
    values = [weekly[k] for k in keys]
    labels = [k.replace("-W", "\nWk ") for k in keys]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(keys))
    bars = ax.bar(x, values, color=PRIMARY, width=0.65)
    ax.set_title("Weekly Job Posting Activity  (Last 12 Weeks)", fontsize=14, fontweight="bold", pad=14)
    ax.set_ylabel("Postings", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(GRID)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=8, color=TEXT)
    fig.tight_layout()
    save(fig, "10_weekly_posting_activity.png")


# ── Chart 11: Top Companies — Postings vs Views (bubble-style bar pair) ──────
def chart_company_postings_vs_views(rows):
    co_posts = Counter(r["_co_en"] for r in rows)
    co_views = defaultdict(int)
    for r in rows:
        co_views[r["_co_en"]] += r["_views"]

    top = [k for k, _ in co_posts.most_common(12)]
    posts = [co_posts[k] for k in top]
    avg_views = [round(co_views[k] / co_posts[k]) for k in top]

    x = np.arange(len(top))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax2 = ax1.twinx()
    ax2.set_facecolor(BG)

    b1 = ax1.bar(x - width / 2, posts, width, label="Job Postings", color=PRIMARY)
    b2 = ax2.bar(x + width / 2, avg_views, width, label="Avg Views / Job", color=ACCENT)

    ax1.set_title("Top 12 Companies — Postings Volume vs. Avg. Candidate Interest",
                  fontsize=13, fontweight="bold", pad=14)
    ax1.set_ylabel("Number of Postings", fontsize=11, color=PRIMARY)
    ax2.set_ylabel("Avg. Views per Posting", fontsize=11, color=ACCENT)
    ax1.set_xticks(x)
    ax1.set_xticklabels(top, rotation=30, ha="right", fontsize=8)
    ax1.tick_params(axis="y", labelcolor=PRIMARY)
    ax2.tick_params(axis="y", labelcolor=ACCENT)
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax1.spines["bottom"].set_visible(True)
    ax1.spines["bottom"].set_color(GRID)
    ax1.grid(False)
    ax2.grid(False)

    lines = [b1, b2]
    labels = ["Job Postings", "Avg Views / Job"]
    ax1.legend(lines, labels, fontsize=10, loc="upper right")
    fig.tight_layout()
    save(fig, "11_postings_vs_candidate_interest.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    rows = load_data()
    print(f"  {len(rows)} job postings loaded.\n")

    print("Generating charts...")
    chart_top_companies(rows)
    chart_by_industry(rows)
    chart_monthly_trend(rows)
    chart_career_levels(rows)
    chart_employment_types(rows)
    chart_company_views(rows)
    chart_avg_views_industry(rows)
    chart_geography(rows)
    chart_industry_employment_stack(rows)
    chart_weekly_trend(rows)
    chart_company_postings_vs_views(rows)

    print(f"\nDone. All charts saved to '{OUT_DIR}/'")
