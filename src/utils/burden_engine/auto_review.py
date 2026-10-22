"""Auto mode of the Burden & Fuel Optimiser: candidate plans, the deterministic pick, and the language-model review.

The pick is made by a fixed rule, not by the language model (ASM-19):

  1. candidates at one production: A - lowest expected cost; B - lowest cost + 1.28 sigma (risk-averse);
     C - lowest expected cost among High-certainty plans; T - today's charged blend (when available);
  2. among the optimised candidates that meet every limit, the lowest cost + RULE_K x sigma(cost) (RULE_K = 1, about
     the 84th percentile of the cost), ties to the higher certainty class;
  3. hold today's blend when that plan's expected saving against it is smaller than RULE_K x sigma of the saving
     ("a saving inside the model's error band is not a reason to change the burden").

The language model then reads the candidate table and the rule's pick and returns a JSON review: the candidate it
would choose, reasons, risks and a monitoring plan. Its choice must be one of the candidate ids, and every number it
writes is checked against the numbers it was given; a different choice is shown as a disagreement next to the rule's
pick, never instead of it. "Best interest" is the plant policy below, written here, not left to the model.

Pure code: dictionaries in, strings and dictionaries out. The model call is in ``ui/burden_optimiser/omega_panel.py``.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping, Sequence

RULE_K = 1.0
CERT_RANK = {"High": 3, "Medium": 2, "Low": 1, "Investigate": 0, "Held": 0}
CANDIDATES = (                                     # "low": every certainty except Investigate (GAS-18)
    ("A", "Lowest expected cost", dict(risk_k=0.0, certainty="low")),
    ("B", "Risk-averse: lowest cost + 1.28 sigma", dict(risk_k=1.28, certainty="low")),
    ("C", "High certainty only", dict(risk_k=0.0, certainty="high")),
)

POLICY = """\
PLANT POLICY (Evonith BF-2) - what "in the plant's best interest" means for this choice:
1. A plan that breaks a limit is not a candidate.
2. Minimise ore + flux + fuel cost per tonne of hot metal, judged at cost + 1 sigma, not at the expected cost alone.
3. Keep today's charged blend when the saving of a change is smaller than one sigma of that saving.
4. A Low-certainty plan (blend or production outside what BF-2 has run, or sigma(coke) above 5 kg/tHM) is chosen only
   if its saving exceeds two sigma of the saving and the change can be made in steps one descent time apart.
5. Among plans within one sigma of each other, prefer the smaller change from today's blend and the higher certainty.
6. Furnace stability (permeability, thermal state) outranks a saving of less than Rs 50/tHM.
7. An Investigate plan (pellet above 25 % of ferrous or sigma(coke) above 8 kg/tHM: beyond common practice) is never
   picked; it is a trial proposal, not an operating plan."""

REVIEW_SYSTEM = """\
You review the choice between candidate burden plans for Evonith Steel BF-2 for the operating team. You are given the \
plant policy, each candidate's cost, coke rate, their 1-sigma uncertainty and certainty class, the shaft-efficiency \
terms behind them, and the plan a fixed rule picked.

Use ONLY the numbers supplied; do not compute new quantities except differences of numbers in the table. Write in \
plain engineering English.

Return ONLY a JSON object, no other text, with exactly these keys:
{"choice": "<one candidate id from the table>",
 "agrees_with_rule": true or false,
 "reasons": ["<= 4 short items: why this candidate, quoting the table"],
 "risks": ["<= 3 short items: what could make it wrong"],
 "monitoring": ["<= 4 short items: what to check after the change, and when"],
 "summary": "<one sentence for the shift in-charge>"}"""


def _f(x: Any, spec: str = ",.0f") -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(v) else format(v, spec)


def candidate_row(cid: str, label: str, r: Mapping[str, Any] | None, today: Mapping[str, Any] | None = None,
                  sig_vs_today: Mapping[str, float] | None = None) -> dict[str, Any]:
    if r is None:
        return dict(id=cid, label=label, available=False)
    om, c, o = r.get("omega") or {}, r["cost"], r["out"]
    row = dict(id=cid, label=label, available=True, feasible=bool(r.get("feasible", False)),
               cost=float(c["total"]), sigma_cost=float(c.get("sigma", 0.0)),
               cost_plus_sigma=float(c["total"]) + RULE_K * float(c.get("sigma", 0.0)),
               coke_dry=float(o["K"]), sigma_coke=float(om.get("sigma_K", 0.0)), omega=float(om.get("value", o.get("omega", math.nan))),
               certainty=om.get("certainty", "Held"), support=om.get("support", ""), outside=list(om.get("outside", [])),
               shares={k: round(float(v), 1) for k, v in r["share_wet"].items() if v > 0.05},
               P_tday=float(r["P_tday"]))
    if today is not None and cid != "T":
        row["saving_vs_today"] = float(today["cost"]["total"]) - row["cost"]
        row["sigma_saving"] = float((sig_vs_today or {}).get("sigma_cost", math.nan))
    return row


def pick(rows: Sequence[Mapping[str, Any]], k: float = RULE_K) -> tuple[str | None, str]:
    """(candidate id, reason) by the fixed rule."""
    ok = [r for r in rows if r.get("available") and r.get("feasible") and r["id"] != "T"
          and r.get("certainty") != "Investigate"]
    if not ok:
        today = next((r for r in rows if r["id"] == "T" and r.get("available")), None)
        return (("T", "No optimised candidate meets every limit; today's blend is kept.") if today
                else (None, "No candidate meets every limit."))
    best = min(ok, key=lambda r: (r["cost"] + k * r["sigma_cost"], -CERT_RANK.get(r["certainty"], 0)))
    today = next((r for r in rows if r["id"] == "T" and r.get("available")), None)
    if today is not None and "saving_vs_today" in best:
        sv, ss = best["saving_vs_today"], best.get("sigma_saving", math.nan)
        if math.isfinite(ss) and sv < k * ss:
            return "T", (f"Best plan {best['id']} saves Rs {sv:,.0f}/tHM against today's blend, less than "
                         f"{k:g} sigma of the saving (Rs {ss:,.0f}/tHM): today's blend is kept.")
    return best["id"], (f"Lowest cost + {k:g} sigma among the plans that meet every limit: {best['id']} at "
                        f"Rs {best['cost_plus_sigma']:,.0f}/tHM ({best['certainty']} certainty).")


def review_context(rows: Sequence[Mapping[str, Any]], rule_pick: str | None, rule_reason: str, *,
                   names: Mapping[str, str], mode_label: str, calibration: Mapping[str, Any] | None = None) -> str:
    cal = dict(calibration or {})
    lines = [POLICY, "", f"RUN: {mode_label}; calibration {cal.get('source', 'n/a')} {cal.get('window', '')}, "
             f"shaft efficiency at the calibration {_f(cal.get('omega'), '.3f')}.", "",
             "CANDIDATES (cost Rs/tHM ore + flux + fuel; coke kg/tHM dry; sigma = 1 standard deviation from the "
             "shaft-efficiency model, relative to the calibration; saving = today's cost minus the candidate's):"]
    for r in rows:
        if not r.get("available"):
            lines.append(f"- {r['id']} {r['label']}: not available")
            continue
        sh = "; ".join(f"{names.get(k, k)} {v:.1f} %" for k, v in sorted(r["shares"].items(), key=lambda kv: -kv[1]))
        txt = (f"- {r['id']} {r['label']}: {'meets every limit' if r['feasible'] else 'BREAKS A LIMIT'}; cost "
               f"{_f(r['cost'])} +/- {_f(r['sigma_cost'])}, cost + 1 sigma {_f(r['cost_plus_sigma'])}; coke "
               f"{_f(r['coke_dry'], '.1f')} +/- {_f(r['sigma_coke'], '.1f')}; shaft efficiency {_f(r['omega'], '.3f')}; "
               f"certainty {r['certainty']}")
        if r.get("outside"):
            txt += " (outside what BF-2 ran: " + "; ".join(r["outside"]) + ")"
        if "saving_vs_today" in r:
            txt += f"; saving against today {_f(r['saving_vs_today'], '+,.0f')} +/- {_f(r['sigma_saving'])}"
        txt += f"; production {_f(r['P_tday'])} t/d; blend {sh}"
        lines.append(txt)
    lines += ["", f"RULE PICK: {rule_pick or 'none'} - {rule_reason}"]
    return "\n".join(lines)


def parse_review(text: str, ids: Sequence[str]) -> tuple[dict[str, Any] | None, list[str]]:
    """The model's JSON (code fences tolerated) and the problems found in it."""
    problems: list[str] = []
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    m = re.search(r"\{.*\}", t, flags=re.S)
    if not m:
        return None, ["no JSON object in the reply"]
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        return None, [f"reply is not valid JSON ({exc.msg})"]
    for key in ("choice", "reasons", "risks", "monitoring", "summary"):
        if key not in d:
            problems.append(f"missing key {key!r}")
    if d.get("choice") not in ids:
        problems.append(f"choice {d.get('choice')!r} is not one of the candidates {list(ids)}")
    for key in ("reasons", "risks", "monitoring"):
        if key in d and not isinstance(d[key], list):
            d[key] = [str(d[key])]
    return d, problems


_NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> list[float]:
    out = []
    for s in _NUM.findall(text or ""):
        try:
            out.append(float(s.replace(",", "").replace("+", "")))
        except ValueError:
            pass
    return out


def ungrounded_numbers(review: Mapping[str, Any], context: str, rel: float = 0.015, abs_tol: float = 0.06) -> list[str]:
    """Numbers in the review that are neither in the context (within rounding) nor a difference of two of them."""
    given = sorted(set(abs(x) for x in _numbers(context)))
    diffs = None
    bad = []
    text = " ".join([str(review.get("summary", ""))] + [str(x) for k in ("reasons", "risks", "monitoring")
                                                          for x in review.get(k, []) or []])
    for x in _numbers(text):
        v = abs(x)
        if v <= 10 and float(v).is_integer():          # counts, steps, hours, ids of limits
            continue
        if any(abs(v - g) <= max(abs_tol, rel * g) for g in given):
            continue
        if diffs is None:
            big = [g for g in given if g >= 1][:400]
            diffs = sorted(set(abs(a - b) for i, a in enumerate(big) for b in big[i + 1:]))
        if any(abs(v - g) <= max(abs_tol, rel * max(g, 1.0)) for g in diffs):
            continue
        bad.append(_fmt_num(x))
    return bad


def _fmt_num(x: float) -> str:
    return f"{x:g}"
