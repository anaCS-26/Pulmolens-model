"""End-to-end LLM evals against a running PulmoLens backend.

Usage (PowerShell, from repo root):
    cd backend
    uvicorn api:app --reload                    # terminal 1
    python -m evals.run_evals                   # terminal 2
    python -m evals.run_evals --case high_confidence_pneumonia
    python -m evals.run_evals --no-judge        # skip Gemma 4 judge layer

Env:
    EVAL_API_BASE   default http://127.0.0.1:8000
    JUDGE_MODEL     default gemma-4-31b-it
"""
import argparse
import base64
import io
import json
import os
import sys
from typing import Dict, List, Tuple

import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw

load_dotenv()

from .asserts import run_keyword_asserts
from .cases import CASES, EvalCase


API_BASE = os.getenv("EVAL_API_BASE", "http://127.0.0.1:8000")


def make_fake_overlay_b64() -> str:
    """Synthetic stand-in for a Grad-CAM-on-X-ray composite.

    Stylised PA chest silhouette with two lung fields, a central cardiac
    shadow, and a hot blob in the right mid-zone. Lets the judge make a
    plausible anatomical region call instead of "red circle on black".
    """
    W = H = 384
    img = Image.new("RGB", (W, H), (8, 8, 12))
    draw = ImageDraw.Draw(img)
    # thoracic outline
    draw.rectangle([60, 60, W - 60, H - 40], fill=(40, 40, 48))
    # left lung field (viewer's right)
    draw.ellipse([80, 90, 175, 320], fill=(120, 120, 130))
    # right lung field (viewer's left)
    draw.ellipse([W - 175, 90, W - 80, 320], fill=(120, 120, 130))
    # cardiac silhouette
    draw.ellipse([155, 170, 245, 320], fill=(70, 70, 80))
    # Grad-CAM hot blob — right mid-zone (viewer's left)
    draw.ellipse([95, 165, 165, 235], fill=(220, 60, 40))
    draw.ellipse([110, 180, 150, 220], fill=(255, 220, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def call_summarize(predictions: Dict[str, float], overlay_b64: str) -> Tuple[str, List[str]]:
    r = requests.post(
        f"{API_BASE}/summarize",
        json={"predictions": predictions, "attention_overlay": overlay_b64},
        stream=True,
        timeout=180,
    )
    r.raise_for_status()
    parts: List[str] = []
    sources: List[str] = []
    for line in r.iter_lines():
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "report" in obj:
            parts.append(obj["report"])
        if "sources" in obj:
            sources = obj["sources"]
    return "".join(parts), sources


def run_case(case: EvalCase, overlay_b64: str, use_judge: bool) -> dict:
    print(f"\n{'='*72}\n>> {case.name}\n{'='*72}")
    if case.notes:
        print(f"   note: {case.notes}")

    report, sources = call_summarize(case.predictions, overlay_b64)

    preview = report.replace("\n", "\n   ")
    if len(preview) > 700:
        preview = preview[:700] + "..."
    safe_preview = preview.encode("ascii", "replace").decode("ascii")
    print(f"   ---- report ({len(report)} chars) ----\n   {safe_preview}")
    if sources:
        print(f"   sources: {sources}")

    asserts = run_keyword_asserts(case, report)
    passed = sum(1 for _, ok, _ in asserts if ok)
    failed = [(name, why) for name, ok, why in asserts if not ok]
    print(f"   ---- keyword asserts: {passed}/{len(asserts)} passed ----")
    for name, why in failed:
        print(f"     FAIL {name} -- {why}")

    judge_result = None
    judge_failed = False
    if use_judge and not case.expect_short_circuit:
        from .judge import judge_hard_failed, judge_report
        print("   ---- invoking judge ----")
        try:
            judge_result = judge_report(
                report=report,
                primary_finding=case.primary_finding,
                predictions=case.predictions,
                overlay_b64=overlay_b64,
                urgent=case.urgent,
            )
            judge_failed = judge_hard_failed(judge_result)
            indented = json.dumps(judge_result, indent=2).replace("\n", "\n     ")
            print(f"     {indented}")
        except Exception as e:
            judge_result = {"_invoke_error": str(e)}
            print(f"     judge invocation failed: {e}")

    return {
        "case": case.name,
        "asserts_passed": passed,
        "asserts_total": len(asserts),
        "asserts_failed": failed,
        "judge": judge_result,
        "judge_failed": judge_failed,
        "report_len": len(report),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="run a single case by name")
    parser.add_argument("--no-judge", action="store_true", help="skip Gemma 4 judge layer")
    args = parser.parse_args()

    use_judge = not args.no_judge
    overlay = make_fake_overlay_b64()

    cases = [c for c in CASES if not args.case or c.name == args.case]
    if not cases:
        print(f"no case named {args.case!r}", file=sys.stderr)
        return 2

    results = [run_case(c, overlay, use_judge) for c in cases]

    print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
    hard_fail = 0
    for r in results:
        kw_ok = r["asserts_passed"] == r["asserts_total"]
        flag = "PASS" if kw_ok and not r["judge_failed"] else "FAIL"
        if not kw_ok or r["judge_failed"]:
            hard_fail += 1
        judge_tag = " [JUDGE FLAGGED]" if r["judge_failed"] else ""
        print(f"  {flag}  {r['case']:<35} asserts {r['asserts_passed']}/{r['asserts_total']}{judge_tag}")

    print(f"\n{len(results) - hard_fail}/{len(results)} cases clean")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
