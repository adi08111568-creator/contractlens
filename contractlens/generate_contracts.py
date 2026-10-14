#!/usr/bin/env python3
"""Generate synthetic healthcare provider contracts plus a ground-truth answer key.

All parties are fictional. Contracts come in five deliberately different layouts
(formal articles, rate-schedule tables, plain-English letter, definitions plus an
amendment, abbreviated term sheet) so that position-based rules break while a
semantic extractor (an LLM) still works - the same problem AArete described.

Usage:
    python generate_contracts.py [--out data] [--n 30] [--seed 42]

Outputs (under --out):
    contracts/pdf/contract_XX.pdf
    contracts/txt/contract_XX.txt
    ground_truth.json   (answer key + layout metadata)
    ground_truth.csv    (answer key only)
"""
import argparse
import csv
import json
import random
import textwrap
from datetime import date, timedelta
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# --------------------------------------------------------------------------
# Fictional parties and reference data
# --------------------------------------------------------------------------
PAYERS = [
    "Meridian Health Plan", "Crestline Health Partners", "Northbridge Assurance Company",
    "Harborview Health Plan", "Summit Care Alliance", "Lakeshore Health Insurance Co.",
    "Pinecrest Medical Plan", "Evergreen Health Partners", "Cobalt Mutual Health",
    "Redwood Community Health Plan",
]
PROVIDERS = [
    "St. Aldric Regional Medical Center", "Riverbend Community Hospital",
    "Oakhaven Orthopedic Group", "Westfield Cardiology Associates",
    "Marlow Children's Hospital", "Brightwater Surgical Center",
    "Northgate Family Physicians", "Kestrel Imaging and Diagnostics",
    "Hollis County General Hospital", "Ridgeway Rehabilitation Institute",
    "Larkspur Women's Health Center", "Tamarack Oncology Partners",
]
STATES = [
    "Texas", "Ohio", "Georgia", "Illinois", "Pennsylvania", "Arizona",
    "Colorado", "Florida", "North Carolina", "Michigan", "Washington", "Tennessee",
]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()

FIELDS = [
    "payer_name", "provider_name", "effective_date", "expiry_date", "auto_renewal",
    "termination_notice_days", "inpatient_per_diem_usd", "outpatient_pct_of_medicare",
    "clean_claim_payment_days", "timely_filing_days", "credentialing_sla_days",
    "governing_law",
]
LAYOUT_NAMES = {0: "A_formal_articles", 1: "B_rate_tables", 2: "C_plain_letter",
                3: "D_definitions_amendment", 4: "E_term_sheet"}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def words(n):
    """Spell out a non-negative integer below 10,000."""
    if n < 20:
        return ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return TENS[t] + ("-" + ONES[o] if o else "")
    if n < 1000:
        h, r = divmod(n, 100)
        return ONES[h] + " hundred" + (" " + words(r) if r else "")
    th, r = divmod(n, 1000)
    return ONES[th] + " thousand" + (" " + words(r) if r else "")


def spelled(n):
    return f"{words(n)} ({n})"


def ordinal(n):
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def fmt_date(d, style):
    if style == "iso":
        return d.isoformat()
    if style == "long":
        return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"
    if style == "legal":
        return f"{ordinal(d.day)} day of {MONTHS[d.month - 1]}, {d.year}"
    if style == "short":
        return f"{d.day:02d} {MONTHS[d.month - 1][:3]} {d.year}"
    raise ValueError(style)


def add_years(d, years):
    try:
        return d.replace(year=d.year + years)
    except ValueError:  # Feb 29
        return d.replace(year=d.year + years, day=28)


def money(n):
    return f"${n:,}"


# --------------------------------------------------------------------------
# Spec generation
# --------------------------------------------------------------------------
def make_spec(rng, used_pairs):
    while True:
        payer, provider = rng.choice(PAYERS), rng.choice(PROVIDERS)
        if (payer, provider) not in used_pairs:
            used_pairs.add((payer, provider))
            break
    eff = date(rng.randint(2021, 2025), rng.randint(1, 12), rng.randint(1, 28))
    term = rng.choice([2, 3, 4, 5])
    exp = add_years(eff, term) - timedelta(days=1)

    notice = rng.choice([60, 90, 120, 180])
    per_diem = rng.randrange(1800, 4225, 25)
    pct = rng.randrange(95, 165, 5)
    clean = rng.choice([15, 20, 30, 45])
    timely = rng.choice([90, 120, 180, 365])
    cred = rng.choice([30, 45, 60, 90])

    truth = {
        "payer_name": payer,
        "provider_name": provider,
        "effective_date": eff.isoformat(),
        "expiry_date": exp.isoformat(),
        "auto_renewal": rng.random() < 0.6,
        "termination_notice_days": notice,
        "inpatient_per_diem_usd": per_diem,
        "outpatient_pct_of_medicare": pct,
        "clean_claim_payment_days": clean,
        "timely_filing_days": timely,
        "credentialing_sla_days": cred,
        "governing_law": rng.choice(STATES),
    }
    # Distractors: plausible numbers that must NOT be extracted for the fields above.
    aux = {
        "term_years": term,
        "nonrenew_days": rng.choice([x for x in (30, 60, 90, 120) if x != notice]),
        "appeals_days": rng.choice([x for x in (60, 150, 210, 270) if x != timely]),
        "dispute_days": rng.choice([30, 45]),
        "bh_per_diem": per_diem + rng.choice([-400, -300, 300, 400]),
        "er_pct": pct + rng.choice([-10, 10, 15]),
        "old_clean_days": clean + 15,
        "old_per_diem": per_diem - rng.choice([100, 150, 200]),
        "prior_date": add_years(eff, -3),
        "base_per_diem": per_diem - rng.choice([150, 200, 250]),
        "base_pct": pct - rng.choice([5, 10]),
        "amend_date": add_years(eff, 1),
        "liability_limit": rng.choice(["$1,000,000", "$2,000,000"]),
        "retention_years": rng.choice([7, 10]),
    }
    return truth, aux


def d_(truth, key):
    return date.fromisoformat(truth[key])


# --------------------------------------------------------------------------
# Layouts. Each returns a list of blocks:
#   ("title", text) ("h", text) ("p", text) ("table", header, rows)
# --------------------------------------------------------------------------
def boilerplate(a, style="formal"):
    return [
        ("p", "Confidentiality. Each party shall keep the financial terms of this Agreement "
              "confidential and shall disclose them only as required by law."),
        ("p", f"Insurance. Provider shall maintain professional liability insurance of not less "
              f"than {a['liability_limit']} per occurrence."),
        ("p", f"Records. Each party shall retain records relating to services for {a['retention_years']} "
              f"years following the date of service."),
        ("p", f"Disputes. The parties shall attempt in good faith to resolve any dispute through "
              f"informal negotiation for up to {a['dispute_days']} days before pursuing other remedies."),
    ]


def layout_a(t, a, rng):
    eff, exp = d_(t, "effective_date"), d_(t, "expiry_date")
    if t["auto_renewal"]:
        renew = (f"1.2 Renewal. Upon expiration of the initial term, this Agreement shall automatically "
                 f"renew for successive one-year terms unless either party gives written notice of "
                 f"non-renewal at least {a['nonrenew_days']} days before the end of the then-current term.")
    else:
        renew = ("1.2 Renewal. This Agreement shall not renew automatically. Any extension beyond the "
                 "initial term requires a written amendment signed by both parties.")
    B = [
        ("title", "PROVIDER PARTICIPATION AGREEMENT"),
        ("p", f'This Provider Participation Agreement (the "Agreement") is entered into as of '
              f'{fmt_date(eff, "long")} (the "Effective Date") by and between {t["payer_name"]} '
              f'("Plan") and {t["provider_name"]} ("Provider").'),
        ("h", "ARTICLE I - TERM AND TERMINATION"),
        ("p", f"1.1 Initial Term. This Agreement commences on the Effective Date and continues for an "
              f"initial term of {a['term_years']} years, expiring on {fmt_date(exp, 'long')}."),
        ("p", renew),
        ("p", f"1.3 Termination Without Cause. Either party may terminate this Agreement without cause "
              f"upon {t['termination_notice_days']} days prior written notice to the other party."),
        ("h", "ARTICLE II - COMPENSATION"),
        ("p", f"2.1 Inpatient Services. Plan shall reimburse Provider for covered acute medical/surgical "
              f"inpatient services at a per diem rate of {money(t['inpatient_per_diem_usd'])} per day. "
              f"Behavioral health inpatient services are reimbursed at {money(a['bh_per_diem'])} per day."),
        ("p", f"2.2 Outpatient Services. Plan shall reimburse Provider for covered general outpatient "
              f"services at {t['outpatient_pct_of_medicare']}% of the then-current Medicare fee schedule. "
              f"Emergency department services are reimbursed at {a['er_pct']}% of the Medicare fee schedule."),
        ("h", "ARTICLE III - CLAIMS"),
        ("p", f"3.1 Payment of Clean Claims. Plan shall pay each clean claim within "
              f"{t['clean_claim_payment_days']} days after receipt."),
        ("p", f"3.2 Timely Filing. Provider shall submit claims within {t['timely_filing_days']} days "
              f"after the date of service. Claim denials may be appealed within {a['appeals_days']} days "
              f"of the denial notice."),
        ("h", "ARTICLE IV - CREDENTIALING"),
        ("p", f"4.1 Plan shall complete credentialing of Provider's practitioners within "
              f"{t['credentialing_sla_days']} days after receipt of a complete application."),
        ("h", "ARTICLE V - GENERAL PROVISIONS"),
        ("p", f"5.1 Governing Law. This Agreement is governed by the laws of the State of "
              f"{t['governing_law']}."),
    ]
    B += boilerplate(a)
    B.append(("p", "IN WITNESS WHEREOF, the parties have executed this Agreement as of the Effective Date."))
    return B


def layout_b(t, a, rng):
    eff, exp = d_(t, "effective_date"), d_(t, "expiry_date")
    renewal = ("Evergreen - renews automatically for successive 12-month periods"
               if t["auto_renewal"] else "Fixed term - no automatic renewal")
    particulars = [
        ["Payer", t["payer_name"]],
        ["Provider", t["provider_name"]],
        ["Effective Date", fmt_date(eff, "short")],
        ["Term", f"{a['term_years']} years"],
        ["Expiration Date", fmt_date(exp, "short")],
        ["Renewal", renewal],
        ["Termination Without Cause", f"{t['termination_notice_days']} days written notice"],
    ]
    rates = [
        ["Inpatient - Acute (Med/Surg)", "Per diem", f"{money(t['inpatient_per_diem_usd'])} / day"],
        ["Inpatient - Behavioral Health", "Per diem", f"{money(a['bh_per_diem'])} / day"],
        ["Outpatient - General", "% of Medicare fee schedule", f"{t['outpatient_pct_of_medicare']}%"],
        ["Emergency Department", "% of Medicare fee schedule", f"{a['er_pct']}%"],
    ]
    admin = [
        ["Clean claim payment", f"Within {t['clean_claim_payment_days']} calendar days of receipt"],
        ["Timely filing", f"{t['timely_filing_days']} days from date of service"],
        ["Appeals window", f"{a['appeals_days']} days from denial notice"],
        ["Credentialing turnaround", f"{t['credentialing_sla_days']} days from complete application"],
        ["Governing law", f"State of {t['governing_law']}"],
    ]
    B = [
        ("title", "PARTICIPATING PROVIDER AGREEMENT - SUMMARY OF TERMS AND RATE SCHEDULE"),
        ("p", "This schedule forms part of the Participating Provider Agreement between the parties "
              "named below and is incorporated by reference."),
        ("h", "SECTION 1 - AGREEMENT PARTICULARS"),
        ("table", ["Item", "Detail"], particulars),
        ("h", "EXHIBIT A - RATE SCHEDULE"),
        ("table", ["Service Category", "Methodology", "Rate"], rates),
        ("h", "EXHIBIT B - ADMINISTRATIVE TERMS"),
        ("table", ["Item", "Term"], admin),
    ]
    B += boilerplate(a)
    return B


def layout_c(t, a, rng):
    eff, exp = d_(t, "effective_date"), d_(t, "expiry_date")
    if t["auto_renewal"]:
        rollover = (f"At the end of that period it will roll over from year to year unless one side gives "
                    f"notice of non-renewal {a['nonrenew_days']} days ahead.")
    else:
        rollover = "It will not roll over; a new signed agreement is needed to continue after that date."
    return [
        ("title", "LETTER AGREEMENT - HOSPITAL SERVICES RATES AND TERMS"),
        ("p", f"Date: {fmt_date(eff, 'legal')}"),
        ("p", f"This letter agreement sets out how {t['payer_name']} (the Plan) will reimburse "
              f"{t['provider_name']} (the Provider) for covered services given to the Plan's members. "
              f"It replaces the parties' earlier arrangement dated {fmt_date(a['prior_date'], 'long')}, "
              f"under which clean claims were paid within {a['old_clean_days']} days and acute inpatient "
              f"care was paid at {money(a['old_per_diem'])} per day. Those earlier terms no longer apply."),
        ("p", f"Rates. Starting on {fmt_date(eff, 'legal')}, the Plan will pay a flat per diem of "
              f"{money(t['inpatient_per_diem_usd'])} for each day of acute medical/surgical inpatient care, "
              f"and {spelled(t['outpatient_pct_of_medicare'])} percent of the Medicare fee schedule for "
              f"general outpatient services. Behavioral health inpatient stays are paid at "
              f"{money(a['bh_per_diem'])} per day, and emergency department visits at "
              f"{a['er_pct']} percent of Medicare."),
        ("p", f"Payment and filing. The Plan will pay each clean claim within "
              f"{spelled(t['clean_claim_payment_days'])} days after it receives the claim. The Provider "
              f"must send claims no later than {spelled(t['timely_filing_days'])} days after the date of "
              f"service. A denied claim may be appealed within {a['appeals_days']} days."),
        ("p", f"Credentialing. The Plan will finish credentialing the Provider's clinicians within "
              f"{spelled(t['credentialing_sla_days'])} days of getting a complete application."),
        ("p", f"Duration. This letter agreement begins on the {fmt_date(eff, 'legal')}, runs for "
              f"{words(a['term_years'])} years, and ends on the {fmt_date(exp, 'legal')}. {rollover} "
              f"Either side may end it early, without cause, by giving {spelled(t['termination_notice_days'])} "
              f"days written notice."),
        ("p", f"Law. This letter agreement is governed by the laws of the State of {t['governing_law']}."),
        ("p", f"Insurance and records. The Provider will carry professional liability cover of "
              f"{a['liability_limit']} per occurrence and keep records for {a['retention_years']} years."),
        ("p", "Please countersign below to confirm that this reflects our agreement."),
    ]


def layout_d(t, a, rng):
    eff, exp = d_(t, "effective_date"), d_(t, "expiry_date")
    if t["auto_renewal"]:
        renew = ("2.2 Upon expiration of the term, this Agreement shall renew automatically for "
                 "successive twelve (12) month terms unless terminated in accordance with Section 2.3.")
    else:
        renew = ("2.2 This Agreement does not renew automatically. Continuation after the expiration "
                 "date requires a signed written amendment.")
    B = [
        ("title", "MASTER PARTICIPATING FACILITY AGREEMENT"),
        ("p", f"This Master Participating Facility Agreement is made on the {fmt_date(eff, 'legal')} "
              f"between {t['payer_name']} and {t['provider_name']}."),
        ("h", "1. DEFINITIONS"),
        ("p", f'"Effective Date" means {fmt_date(eff, "long")}. "Clean Claim" means a claim that can be '
              f'processed without obtaining additional information from the Provider. "Medicare Fee '
              f'Schedule" means the applicable CMS fee schedule in effect on the date of service. '
              f'"Plan" means {t["payer_name"]}. "Facility" means {t["provider_name"]}.'),
        ("h", "2. TERM"),
        ("p", f"2.1 The term of this Agreement begins on the Effective Date and ends on "
              f"{fmt_date(exp, 'short')}, unless sooner terminated."),
        ("p", renew),
        ("p", f"2.3 Either party may terminate this Agreement without cause by giving "
              f"{t['termination_notice_days']} days written notice."),
        ("h", "3. COMPENSATION"),
        ("p", f"3.1 Acute inpatient services shall be reimbursed at a per diem of "
              f"{money(a['base_per_diem'])}."),
        ("p", f"3.2 General outpatient services shall be reimbursed at {a['base_pct']}% of the Medicare "
              f"Fee Schedule."),
        ("h", "4. CLAIMS"),
        ("p", f"4.1 The Plan shall pay each Clean Claim within {t['clean_claim_payment_days']} days of "
              f"receipt. 4.2 The Facility shall submit claims within {t['timely_filing_days']} days of the "
              f"date of service."),
        ("h", "5. CREDENTIALING"),
        ("p", f"5.1 The Plan shall complete credentialing within {t['credentialing_sla_days']} days of "
              f"a complete application."),
        ("h", "6. GENERAL"),
        ("p", f"6.1 This Agreement is governed by the laws of the State of {t['governing_law']}."),
    ]
    B += boilerplate(a)
    B += [
        ("h", "AMENDMENT NO. 1"),
        ("p", f"Effective {fmt_date(a['amend_date'], 'long')}, the parties amend Sections 3.1 and 3.2 of "
              f"the Agreement as set out below. All other terms remain unchanged."),
        ("p", f"Section 3.1 is deleted and replaced with: \"Acute inpatient services shall be reimbursed "
              f"at a per diem of {money(t['inpatient_per_diem_usd'])}.\""),
        ("p", f"Section 3.2 is deleted and replaced with: \"General outpatient services shall be "
              f"reimbursed at {t['outpatient_pct_of_medicare']}% of the Medicare Fee Schedule.\""),
    ]
    return B


def layout_e(t, a, rng):
    eff, exp = d_(t, "effective_date"), d_(t, "expiry_date")
    rows = [
        ["Plan (payer)", t["payer_name"]],
        ["Provider", t["provider_name"]],
        ["Eff. date", fmt_date(eff, "short")],
        ["Term", f"{a['term_years']} yrs"],
        ["Exp. date", fmt_date(exp, "short")],
        ["Auto-renew", "Y - 1-yr periods" if t["auto_renewal"] else "N"],
        ["Term. w/o cause", f"{t['termination_notice_days']}d written notice"],
        ["IP acute per diem", f"{money(t['inpatient_per_diem_usd'])}"],
        ["OP general", f"{t['outpatient_pct_of_medicare']}% of MCR FS"],
        ["CCP", f"{t['clean_claim_payment_days']}d from receipt"],
        ["Timely filing", f"{t['timely_filing_days']}d from DOS"],
        ["Cred. SLA", f"{t['credentialing_sla_days']}d from complete app"],
        ["Gov. law", f"State of {t['governing_law']}"],
    ]
    return [
        ("title", "TERM SHEET - PARTICIPATION AGREEMENT"),
        ("table", ["Item", "Term"], rows),
        ("p", "Abbreviations: MCR FS = Medicare fee schedule; CCP = clean claim payment; "
              "DOS = date of service; IP = inpatient; OP = outpatient."),
        ("p", f"Not covered by the key terms above: behavioral health IP at {money(a['bh_per_diem'])} "
              f"per diem; ED at {a['er_pct']}% of MCR FS; appeals within {a['appeals_days']}d of denial."),
        ("p", f"Prof. liability cover {a['liability_limit']} per occurrence; records retained "
              f"{a['retention_years']} yrs."),
    ]


LAYOUTS = [layout_a, layout_b, layout_c, layout_d, layout_e]


# --------------------------------------------------------------------------
# Renderers
# --------------------------------------------------------------------------
def render_txt(blocks, width=88):
    out = []
    for blk in blocks:
        kind = blk[0]
        if kind == "title":
            out += [blk[1].center(width).rstrip(), "=" * width, ""]
        elif kind == "h":
            out += ["", blk[1], ""]
        elif kind == "p":
            out += [textwrap.fill(blk[1], width, break_on_hyphens=False), ""]
        elif kind == "table":
            header, rows = blk[1], blk[2]
            ncol = len(header)
            widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(ncol)]
            while sum(widths) + 3 * (ncol - 1) > width:  # shrink the widest column to fit
                widths[widths.index(max(widths))] -= 1

            def fmt_row(row):
                cells = [textwrap.wrap(str(c), widths[i]) or [""] for i, c in enumerate(row)]
                height = max(len(c) for c in cells)
                lines = []
                for k in range(height):
                    lines.append(" | ".join((cells[i][k] if k < len(cells[i]) else "").ljust(widths[i])
                                            for i in range(ncol)).rstrip())
                return lines

            sep = "-+-".join("-" * w for w in widths)
            out += fmt_row(header) + [sep]
            for r in rows:
                out += fmt_row(r)
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def render_pdf(blocks, path):
    title = ParagraphStyle("t", fontName="Times-Bold", fontSize=15, leading=19,
                           alignment=TA_CENTER, spaceAfter=14)
    head = ParagraphStyle("h", fontName="Times-Bold", fontSize=11, leading=14,
                          spaceBefore=10, spaceAfter=5, keepWithNext=1)
    body = ParagraphStyle("p", fontName="Times-Roman", fontSize=10.5, leading=14,
                          alignment=TA_JUSTIFY, spaceAfter=6)
    cell = ParagraphStyle("c", fontName="Times-Roman", fontSize=9.5, leading=12)
    cell_b = ParagraphStyle("cb", parent=cell, fontName="Times-Bold")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.grey)
        canvas.drawString(inch, 0.55 * inch,
                          "Synthetic sample contract for demonstration only - all parties are fictional.")
        canvas.drawRightString(LETTER[0] - inch, 0.55 * inch, f"Page {doc.page}")
        canvas.restoreState()

    story = []
    for blk in blocks:
        kind = blk[0]
        if kind == "title":
            story.append(Paragraph(escape(blk[1]), title))
        elif kind == "h":
            story.append(Paragraph(escape(blk[1]), head))
        elif kind == "p":
            story.append(Paragraph(escape(blk[1]), body))
        elif kind == "table":
            header, rows = blk[1], blk[2]
            data = [[Paragraph(escape(str(c)), cell_b) for c in header]]
            data += [[Paragraph(escape(str(c)), cell) for c in r] for r in rows]
            avail = LETTER[0] - 2 * inch
            if len(header) == 3:
                cols = [avail * 0.38, avail * 0.38, avail * 0.24]
            else:
                cols = [avail * 0.32, avail * 0.68]
            tbl = Table(data, colWidths=cols, repeatRows=1)
            tbl.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EDF2")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story += [tbl, Spacer(1, 8)]
    doc = SimpleDocTemplate(str(path), pagesize=LETTER, leftMargin=inch, rightMargin=inch,
                            topMargin=inch, bottomMargin=inch,
                            title="Synthetic provider contract", author="ContractLens demo")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    (out / "contracts" / "pdf").mkdir(parents=True, exist_ok=True)
    (out / "contracts" / "txt").mkdir(parents=True, exist_ok=True)

    layout_ids = [i % len(LAYOUTS) for i in range(args.n)]
    rng.shuffle(layout_ids)  # so file order does not reveal the layout

    used, key = set(), {}
    for i, lid in enumerate(layout_ids, start=1):
        truth, aux = make_spec(rng, used)
        blocks = LAYOUTS[lid](truth, aux, rng)
        name = f"contract_{i:02d}"
        (out / "contracts" / "txt" / f"{name}.txt").write_text(render_txt(blocks), encoding="utf-8")
        render_pdf(blocks, out / "contracts" / "pdf" / f"{name}.pdf")
        key[name] = {"layout": LAYOUT_NAMES[lid], "truth": truth}

    (out / "ground_truth.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    with open(out / "ground_truth.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["contract_id", "layout"] + FIELDS)
        for name, rec in key.items():
            w.writerow([name, rec["layout"]] + [rec["truth"][k] for k in FIELDS])

    print(f"Generated {args.n} contracts in {out.resolve()}")
    counts = {}
    for rec in key.values():
        counts[rec["layout"]] = counts.get(rec["layout"], 0) + 1
    for layout, c in sorted(counts.items()):
        print(f"  {layout}: {c}")


if __name__ == "__main__":
    main()
