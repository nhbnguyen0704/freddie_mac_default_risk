import json
import logging
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from helpers.config import settings

log = logging.getLogger('label')

DPD_BUCKETS: Dict[int, str] = {
    1: "30-59 days past due",
    2: "60-89 days past due",
    3: "90-119 days past due",
    4: "120-149 days past due",
    5: "150-179 days past due",
    6: "180+ days past due",
}

@dataclass(frozen=True)
class LabelSpec:
    """The bad definition: DPD threshold, performance window, and how credit events, workouts and payoffs count."""
    name: str
    dpd_threshold: int
    window_months: int
    basis: str
    include_credit_event: bool
    prepay: str
    provisional: bool
    rationale: str
    include_special_mention: bool = False

    @property
    def dpd_label(self) -> str:
        return DPD_BUCKETS.get(self.dpd_threshold, f'status {self.dpd_threshold}+')

    def describe(self) -> str:
        """The definition in plain words, followed by its rationale."""
        basis = ("reaches the threshold in any month of the window" if self.basis == 'ever' else
                 "is at or beyond the threshold in the final month of the window")
        extra = (" A terminal credit-event disposition (third party sale, short sale or charge off, REO disposition) "
                 "inside the window also counts as bad." if self.include_credit_event else "")
        special = (" A special-mention event (a modification, a payment deferral, a forbearance or repayment plan, "
                   "or principal made non-interest-bearing) also counts as bad, because a workout resets the "
                   "delinquency clock and would otherwise be recorded as a cure. Accommodation attributed to a "
                   "declared disaster is excluded."
                   if self.include_special_mention else "")
        prepay = (" A loan that pays off inside the window without going bad is labelled good."
                  if self.prepay == 'good' else
                  " A loan that pays off inside the window is censored and dropped")
        return (f"An observation is bad if the loan {basis}, where the threshold is "
                f"{self.dpd_label} ({self.window_months}-month performance window). "
                f"{extra}{special}{prepay} {self.rationale}")

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['description'] = self.describe()
        return d

FROM_EVIDENCE = "rollrate"

FALLBACK_WINDOW = 12

ROLL_RATE_JSON = settings.data_curated / "roll_rate" / "recommended_definition.json"
VINTAGE_JSON = settings.data_curated / "vintage" / "suggested_window.json"

def _read(path: Path, what: str) -> dict:
    """A JSON evidence file, failing clearly when it is missing or malformed."""
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not a valid JSON") from exc

def _pick_window(vt: dict, capture: Optional[float]) -> Tuple[Optional[int], dict, str]:
    """The window in months, the vintage option it came from and an override caveat. With no capture asked for, the
    vintage analysis's recommended window; otherwise the option with that capture share."""
    opts = [o for o in (vt.get("window_options") or []) if o.get("months")]
    if capture is None:
        pick = next((o for o in opts if o.get("recommended")), None)
        months = vt.get("suggested_window_months") or (pick or {}).get("months")
        return (int(months) if months else None), (pick or {}), ""

    match = next((o for o in opts if o.get("share") is not None
                  and abs(float(o["share"]) - capture) < 1e-9), None)
    if match is None:
        flat = vt.get(f"window_{int(round(100 * capture))}pct_capture_months")
        if flat:
            return int(flat), {}, _capture_caveat(vt, capture, int(flat))
        offered = ", ".join(f"{100 * o['share']:.0f}%" for o in opts
                            if o.get("share") is not None) or "none"
        raise ValueError(
            f"The vintage analysis offers no {100 * capture:.0f}% capture window. "
            f"Offered: {offered}. Either pass one of those, or change "
            f"CAPTURE_TARGETS in src/jobs/vintage_main.py and re-run it.")
    return int(match["months"]), match, _capture_caveat(vt, capture, int(match["months"]))


def _capture_caveat(vt: dict, capture: float, months: int) -> str:
    """A warning when the capture asked for differs from the one the vintage analysis recommends; empty otherwise."""
    rec = vt.get("recommended_capture")
    if rec is None or abs(float(rec) - capture) < 1e-9:
        return ""
    rec_months = vt.get("suggested_window_months")
    direction = ("SHORTER" if rec_months and months < int(rec_months)
                 else "LONGER" if rec_months else "DIFFERENT")
    tail = ""
    if direction == "SHORTER":
        tail = (f" Roughly {100 * (1 - capture):.0f}% of a cohort's bads arrive "
                f"after month {months} and will be labelled GOOD.")
    return (f"THE WINDOW IS A DELIBERATE OVERRIDE, {direction} than the "
            f"{100 * float(rec):.0f}% capture the vintage recommends"
            + (f" ({rec_months} months)" if rec_months else "")
            + f": {100 * capture:.0f}% capture was asked for instead.{tail} ")


def from_evidence(roll_rate_json: Optional[Path] = None, vintage_json: Optional[Path] = None,
                  capture: Optional[float] = None) -> LabelSpec:
    """The LabelSpec from the roll-rate recommendation (threshold) and the vintage analysis (window). When the
    vintage analysis has not run, or ran and produced no window, the window falls back to FALLBACK_WINDOW
    months. The spec is provisional then, and also when the roll-rate reading is inconclusive or the window is
    overridden."""
    roll_rate_json = Path(roll_rate_json or ROLL_RATE_JSON)
    vintage_json = Path(vintage_json or VINTAGE_JSON)

    rr = _read(roll_rate_json, "roll-rate recommendation")
    threshold = rr.get('dpd_threshold')
    if not threshold:
        raise ValueError(f"{roll_rate_json} is not a valid roll rate recommendation")
    threshold = int(threshold)

    window, window_note, provisional = FALLBACK_WINDOW, "", True
    overridden = False
    if vintage_json.exists():
        vt = _read(vintage_json, "vintage suggested window")
        horizon = vt.get('full_horizon_months')
        suggested, pick, caveat = _pick_window(vt, capture)
        if suggested:
            window, provisional = int(suggested), False
            overridden = bool(caveat)
            voting = pick.get("cohorts_voting") or vt.get("cohorts_at_window") or 0
            iqr = (f", and the middle half of them put it between "
                   f"{pick['p25']} and {pick['p75']} months"
                   if pick.get("p25") and pick.get("p75") else "")
            got = pick.get("captured")
            if got is None:
                got = vt.get("share_of_horizon_bads_captured")
            window_note = (
                caveat
                + f"The {window}-month window is "
                + (pick.get("criterion") or vt.get("criterion")
                   or "read off the capture curve")
                + f". It is the median over the {voting} cohorts at least that "
                  f"deep, of {vt.get('cohorts') or 0} in the panel{iqr}. By that "
                  f"age {100 * (got or 0):.1f}% of the bads those cohorts "
                  f"eventually show have arrived. "
            )
        else:
            window_note = (
                f"THE WINDOW IS NOT EVIDENCE-BASED: the vintage analysis ran but "
                f"produced no window from its {horizon}-month horizon, so "
                f"{FALLBACK_WINDOW} months is convention standing in. "
            )
    else:
        window_note = (
            f"THE WINDOW IS NOT EVIDENCE-BASED: {vintage_json} is absent, so the "
            f"final vintage analysis has not been run and {FALLBACK_WINDOW} months "
            f"is convention standing in. "
        )

    conclusive = bool(rr.get('conclusive'))

    rr_window = rr.get('window_months')
    obs_half = rr.get("observation_half_months")
    out_half = rr.get("outcome_half_months")
    horizon_note = ''
    if rr_window:
        horizon_note = (f"Read across the two halves of the {rr_window}-month "
                        f"window ({obs_half} months observed, {out_half} "
                        f"predicted): the worst rung a loan reached in the first "
                        f"half against the worst it reached in the second. ")
    by_w = rr.get('by_window') or {}
    if len(by_w) > 1 and not rr.get('stable_across_windows'):
        picks = ", ".join(f"{w}m -> {v.get('state')}" for w, v in by_w.items())
        horizon_note += (f"THE WINDOW LENGTHS DISAGREE ({picks}), so this threshold "
                         f"is the {rr_window}-month reading")
    elif len(by_w) > 1:
        horizon_note += ("Every window length looked at picks the same rung")

    rationale = (
        f"Threshold set by the roll-rate analysis across the halves of the "
        f"performance window: {rr.get('reason', '')}. "
        f"{horizon_note}"
        f"{window_note}"
    )

    if not conclusive:
        rationale = ("THE THRESHOLD IS NOT SETTLED -- the roll-rate analysis "
                     "reported no point of no return and this is its fallback. "
                     + rationale)
        provisional = True
    if overridden:
        provisional = True

    name = f"d{30 * threshold}_{window}m"

    return LabelSpec(name=name, dpd_threshold=threshold, window_months=window,
                     basis="ever", include_credit_event=True, prepay="censor",
                     provisional=provisional, rationale=rationale,
                     include_special_mention=True)

def resolve(roll_rate_json: Optional[Path] = None, vintage_json: Optional[Path] = None,
            capture: Optional[float] = None) -> LabelSpec:
    """The label spec, from the evidence files."""
    return from_evidence(roll_rate_json, vintage_json, capture)

def selectable() -> List[str]:
    return [FROM_EVIDENCE]
