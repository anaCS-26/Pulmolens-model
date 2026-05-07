from typing import List, Tuple

from .cases import EvalCase, PROMPT_LEAK_FORBIDDEN


def run_keyword_asserts(case: EvalCase, report: str) -> List[Tuple[str, bool, str]]:
    results: List[Tuple[str, bool, str]] = []
    lower = report.lower()

    if case.expect_short_circuit:
        for needle in case.must_contain:
            results.append((
                f"contains[{needle}]",
                needle.lower() in lower,
                f"short-circuit message must contain '{needle}'",
            ))
        return results

    results.append((
        "non_empty",
        len(report.strip()) > 80,
        "report must be substantive (>80 chars)",
    ))

    for phrase in PROMPT_LEAK_FORBIDDEN:
        results.append((
            f"no_leak[{phrase}]",
            phrase.lower() not in lower,
            f"prompt scaffolding '{phrase}' must not leak into output",
        ))

    for needle in case.must_contain:
        results.append((
            f"contains[{needle}]",
            needle.lower() in lower,
            f"must contain '{needle}'",
        ))

    for needle in case.must_not_contain:
        results.append((
            f"absent[{needle}]",
            needle.lower() not in lower,
            f"must not contain '{needle}'",
        ))

    return results
