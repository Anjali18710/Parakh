"""Suspicious Transaction Report (STR) as a PDF, laid out like the report Indian banks file with FIU-IND.

The parts follow the structure of an FIU-IND STR: report details, principal officer, the subject
account, the suspicious transactions, grounds of suspicion and action taken. It is a *draft for the
bank's Principal Officer*, not a filing: FIU-IND accepts STRs only through its FINnet portal, and
fields Parakh cannot know (KYC details, branch, officer) are left marked for completion.

Until an analyst confirms the case, every page carries a DRAFT watermark.
"""
from __future__ import annotations

import io
import re
from datetime import datetime, timezone

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from parakh.data.store import TransactionStore

INK = colors.HexColor("#1f2a37")
MUTED = colors.HexColor("#5b6675")
RULE = colors.HexColor("#c9ced6")
GOLD = colors.HexColor("#b8860b")
FLAG = colors.HexColor("#b42318")
SHADE = colors.HexColor("#f3f4f6")

TO_COMPLETE = "To be completed by the reporting entity"

# FIU-IND STRs ask the bank to tick the broad reasons for suspicion; these are mapped from what the agents found.
REASONS = {
    "activity": "Activity in account: unusual for the account's normal pattern",
    "nature": "Nature of transactions: money passed through rather than used",
    "value": "Value of transactions: large amounts moved in a short time",
    "manipulation": "Attempt to influence the review: instructions planted in payment remarks",
}


def rupees(value: float | int | None) -> str:
    """Rs. 1,23,45,678.90 (Indian digit grouping). The PDF fonts have no rupee sign, so 'Rs.' is used."""
    if value is None:
        return "-"
    neg, value = value < 0, abs(float(value))
    whole, frac = f"{value:.2f}".split(".")
    head, tail = whole[:-3], whole[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    text = ",".join(groups + [tail]) if groups else tail
    return f"{'-' if neg else ''}Rs. {text}.{frac}"


def _n(count: int, noun: str) -> str:
    return f"{count:,} {noun}{'' if count == 1 else 's'}"


def _plain(text: str) -> str:
    """Make agent text safe for the PDF: rupee sign to 'Rs.', escape markup."""
    text = (text or "").replace("₹", "Rs. ").replace("Rs.  ", "Rs. ")
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _styles() -> dict[str, ParagraphStyle]:
    base = ParagraphStyle("base", fontName="Helvetica", fontSize=9, leading=12.5, textColor=INK, alignment=TA_LEFT)
    return {
        "base": base,
        "small": ParagraphStyle("small", parent=base, fontSize=7.5, leading=10, textColor=MUTED),
        "cell": ParagraphStyle("cell", parent=base, fontSize=7.5, leading=9.5),
        "title": ParagraphStyle("title", parent=base, fontName="Helvetica-Bold", fontSize=15, leading=19),
        "part": ParagraphStyle("part", parent=base, fontName="Helvetica-Bold", fontSize=10, leading=13,
                               spaceBefore=10, spaceAfter=4, textColor=INK),
        "label": ParagraphStyle("label", parent=base, fontSize=8, textColor=MUTED),
        "flag": ParagraphStyle("flag", parent=base, textColor=FLAG),
    }


def _kv(rows: list[tuple[str, str]], st) -> Table:
    data = [[Paragraph(k, st["label"]), Paragraph(v, st["base"])] for k, v in rows]
    t = Table(data, colWidths=[52 * mm, None])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    return t


def _grid(header: list[str], rows: list[list[str]], widths: list[float], st, right: set[int] = frozenset()) -> Table:
    data = [[Paragraph(f"<b>{h}</b>", st["cell"]) for h in header]]
    for r in rows:
        data.append([Paragraph(c, ParagraphStyle("r", parent=st["cell"], alignment=2) if i in right else st["cell"])
                     for i, c in enumerate(r)])
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), SHADE),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, INK),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def cited_txn_ids(case: dict) -> list[str]:
    ids = [t for c in (case.get("report") or {}).get("claims", []) for t in c.get("txn_ids", [])]
    ids += [i["txn_id"] for i in case.get("injections") or []]
    return list(dict.fromkeys(ids))


def reasons_for(case: dict) -> list[str]:
    report, pattern = case.get("report") or {}, case.get("pattern") or {}
    shape = pattern.get("shape") or {}
    out = []
    if (case.get("alert") or {}).get("signals"):
        out.append("activity")
    if (shape.get("share_passed_through") or 0) >= 0.7 or pattern.get("kind") == "laundering":
        out.append("nature")
    if any((c.get("amount") or 0) >= 1_000_000 for c in report.get("claims", [])):
        out.append("value")
    if case.get("injections"):
        out.append("manipulation")
    return out


def build_str(case: dict, store: TransactionStore, reporting_entity: str) -> bytes:
    """Render the STR for one case and return the PDF bytes."""
    st = _styles()
    report = case.get("report") or {}
    pattern = case.get("pattern") or {}
    verification = case.get("verification") or {}
    decision = case.get("decision") or {}
    account = case["account"]
    final = case.get("status") == "confirmed_suspicious"
    generated = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
    ref = f"PARAKH-STR-{case['_id'][:8].upper()}"

    story: list = []
    story += [Paragraph("Suspicious Transaction Report", st["title"]),
              Paragraph(f"Prepared in the format of an FIU-IND STR &middot; {ref} &middot; generated {generated}",
                        st["small"]),
              Spacer(1, 4)]
    status_line = ("Confirmed by an analyst. Ready for the Principal Officer to review and file on FINnet."
                   if final else "DRAFT. Not yet confirmed by an analyst. Do not file.")
    story.append(Paragraph(f"<b>Status:</b> {status_line}", st["flag"] if not final else st["base"]))

    # Part 1
    story.append(Paragraph("Part 1 &middot; Details of report", st["part"]))
    story.append(_kv([
        ("Reporting entity", _plain(reporting_entity)),
        ("Report reference", ref),
        ("Report type", "Suspicious Transaction Report (STR)"),
        ("Case opened", _plain(str(case.get("created_at", ""))[:19].replace("T", " "))),
        ("Prepared by", "Parakh agentic investigation, reviewed by a human analyst" if final
         else "Parakh agentic investigation (awaiting analyst review)"),
    ], st))

    # Part 2
    story.append(Paragraph("Part 2 &middot; Principal Officer and reporting branch", st["part"]))
    story.append(_kv([("Principal Officer name", TO_COMPLETE), ("Designation and contact", TO_COMPLETE),
                      ("Reporting branch", TO_COMPLETE)], st))

    # Part 3
    story.append(Paragraph("Part 3 &middot; Subject account", st["part"]))
    inc, out = store.in_txns(account), store.out_txns(account)
    allt = store.account_txns(account)
    story.append(_kv([
        ("Account number", account),
        ("Account holder and KYC", TO_COMPLETE),
        ("Activity period in data", f"{allt.timestamp.min():%d %b %Y} to {allt.timestamp.max():%d %b %Y}"
         if len(allt) else "-"),
        ("Money received", f"{rupees(inc.amount.sum())} in {_n(len(inc), 'payment')} from "
                           f"{_n(inc.from_account.nunique(), 'sender')}"),
        ("Money sent", f"{rupees(out.amount.sum())} in {_n(len(out), 'payment')} to "
                       f"{_n(out.to_account.nunique(), 'recipient')}"),
        ("Linked accounts", ", ".join(sorted({a for c in report.get("claims", []) for a in c.get("accounts", [])}
                                             - {account})[:20]) or "-"),
    ], st))

    # Part 4
    story.append(Paragraph("Part 4 &middot; Suspicious transactions", st["part"]))
    ids = cited_txn_ids(case)
    rows = store.get_txns(ids).sort_values("timestamp")
    flagged = {i["txn_id"] for i in case.get("injections") or []}
    tx_rows = [[f"{pd.Timestamp(r.timestamp):%d-%m-%Y %H:%M}", r.txn_id, r.from_account, r.to_account,
                rupees(r.amount), _plain(r.payment_format),
                ("<font color='#b42318'>[see Part 6]</font>" if r.txn_id in flagged else _plain(r.remarks[:40]))]
               for r in rows.itertuples()]
    if tx_rows:
        story.append(_grid(["Date and time", "Txn ref", "From", "To", "Amount", "Mode", "Remark"], tx_rows,
                           [25 * mm, 18 * mm, 19 * mm, 19 * mm, 30 * mm, 20 * mm, None], st, right={4}))
        story.append(Paragraph(f"{len(tx_rows)} transactions totalling {rupees(rows.amount.sum())}.", st["small"]))
    else:
        story.append(Paragraph("No transactions were cited.", st["base"]))

    # Part 5
    gos = [Paragraph("Part 5 &middot; Grounds of suspicion", st["part"]),
           Paragraph(_plain(report.get("summary", "")), st["base"]), Spacer(1, 4)]
    typology = (pattern.get("pattern_id") or report.get("pattern_id") or "none").replace("_", " ")
    gos.append(_kv([
        ("Typology", f"{typology} ({pattern.get('confidence', 0):.0%} confidence)" if pattern else typology),
        ("Agents' verdict", f"{(report.get('verdict') or '-').replace('_', ' ')}, risk {report.get('risk_score', '-')}/100"),
        ("Reasons for suspicion", "<br/>".join(f"&bull; {REASONS[r]}" for r in reasons_for(case)) or "-"),
    ], st))
    story.append(KeepTogether(gos))
    claims = report.get("claims", [])
    results = {r["index"]: r for r in verification.get("claim_results", [])}
    if claims:
        story.append(Spacer(1, 5))
        story.append(_grid(["#", "Finding", "Evidence", "Checked"],
                           [[str(i + 1), _plain(c["statement"]), ", ".join(c.get("txn_ids", [])),
                             "yes" if results.get(i, {}).get("ok", True) else "<font color='#b42318'>no</font>"]
                            for i, c in enumerate(claims)],
                           [8 * mm, None, 45 * mm, 16 * mm], st))
        story.append(Paragraph(
            f"Every finding was recomputed from the transaction records: {verification.get('claims_verified', 0)} "
            f"of {verification.get('claims_total', len(claims))} confirmed.", st["small"]))
    if report.get("checks"):
        story.append(Spacer(1, 4))
        story.append(Paragraph("<b>Also checked, nothing found:</b> " +
                               "; ".join(_plain(c["description"]) for c in report["checks"]), st["base"]))

    # Part 6
    if case.get("injections"):
        story.append(Paragraph("Part 6 &middot; Attempt to influence the review", st["part"]))
        story.append(Paragraph(
            "These payment remarks contain instructions aimed at an automated reviewer. They were withheld "
            "from the AI agents and are reported here because they indicate intent to evade detection.", st["base"]))
        story.append(Spacer(1, 4))
        story.append(_grid(["Txn ref", "From", "To", "Remark as typed", "Why flagged"],
                           [[i["txn_id"], i["from"], i["to"], _plain(i["remark"]), _plain("; ".join(i["reasons"]))]
                            for i in case["injections"]],
                           [18 * mm, 19 * mm, 19 * mm, None, 40 * mm], st))

    # Part 7
    action = [Paragraph(f"Part {7 if case.get('injections') else 6} &middot; Action taken", st["part"])]
    if decision:
        action.append(_kv([
            ("Analyst decision", {"approve": "Agreed with the agents", "reject": "Overruled the agents"}.get(
                decision.get("action"), _plain(decision.get("action", "")))),
            ("Analyst note", _plain(decision.get("note") or "-")),
            ("Decided at", _plain(str(decision.get("at") or case.get("updated_at", ""))[:19].replace("T", " "))),
            ("Recommended next step", _plain(report.get("recommended_action", "-"))),
        ], st))
    else:
        action.append(_kv([("Analyst decision", "Pending"),
                           ("Recommended next step", _plain(report.get("recommended_action", "-")))], st))
    action += [Spacer(1, 10), Paragraph(
        "Prepared with Parakh, an AI-assisted investigation tool. Figures are computed from the bank's records, "
        "not by the AI. This document is a draft for the Principal Officer; STRs are filed with FIU-IND through "
        "FINnet. Contents are confidential: tipping off the customer is prohibited.", st["small"])]
    story.append(KeepTogether(action))

    buf = io.BytesIO()

    def decorate(canvas, doc):
        canvas.saveState()
        w, h = A4
        canvas.setStrokeColor(GOLD)
        canvas.setLineWidth(2)
        canvas.line(18 * mm, h - 12 * mm, w - 18 * mm, h - 12 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 10 * mm, f"{ref} · CONFIDENTIAL")
        canvas.drawRightString(w - 18 * mm, 10 * mm, f"Page {doc.page}")
        if not final:
            canvas.setFont("Helvetica-Bold", 90)
            canvas.setFillColor(colors.Color(0.71, 0.14, 0.09, alpha=0.08))
            canvas.translate(w / 2, h / 2)
            canvas.rotate(40)
            canvas.drawCentredString(0, -30, "DRAFT")
        canvas.restoreState()

    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm,
                            bottomMargin=18 * mm, title=f"STR {account}", author="Parakh",
                            subject="Suspicious Transaction Report (draft)")
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    return buf.getvalue()


def filename(case: dict) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "", case["account"])
    return f"STR_{safe}_{'final' if case.get('status') == 'confirmed_suspicious' else 'draft'}.pdf"
