"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_REF_LINE = re.compile(r"^\s*\[(\d+)\]\s*(.*)$")


def _group_numbers(group):
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            numbers.extend(range(a, b + 1) if 0 <= b - a <= 200 else [a, b])
        else:
            numbers.append(int(part))
    return numbers


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK)."""
    problems = []

    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json"]

    seen_urls = set()
    by_n = {}
    for entry in sources:
        if not isinstance(entry, dict):
            problems.append(f"invalid source entry: {entry!r}")
            continue
        n = entry.get("n")
        if not isinstance(n, int):
            problems.append(f"source entry has invalid n: {n!r} (must be int)")
        elif n in by_n:
            problems.append(f"duplicate source number in sources.json: [{n}]")
        else:
            by_n[n] = entry

        url = entry.get("url")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            problems.append(f"source [{n}] URL must start with http:// or https://: {url!r}")
        elif url in seen_urls:
            problems.append(f"duplicate URL in sources.json: {url}")
        else:
            seen_urls.add(url)

    ref_matches = list(_REF_HEADING.finditer(report_text))
    if not ref_matches:
        problems.append("missing '## References' heading")
        body = report_text
        ref_section = ""
    else:
        last_match = ref_matches[-1]
        body = report_text[:last_match.start()]
        ref_section = report_text[last_match.end():]

    # Extract citations in body only (ignoring code blocks/spans and links)
    segments = _CODE.split(body)
    cited = set()
    for i, segment in enumerate(segments):
        if i % 2:
            continue
        for match in _GROUP.finditer(segment):
            for num in _group_numbers(match.group(1)):
                cited.add(num)

    source_ns = set(by_n.keys())
    for num in sorted(cited):
        if num not in source_ns:
            problems.append(f"[{num}] cited in body but missing from sources.json")
    for num in sorted(source_ns):
        if num not in cited:
            problems.append(f"source [{num}] never cited in body")

    if ref_matches:
        ref_line_counts = {}
        ref_lines_by_n = {}
        for line in ref_section.splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            m = _REF_LINE.match(line_str)
            if m:
                n_val = int(m.group(1))
                ref_line_counts[n_val] = ref_line_counts.get(n_val, 0) + 1
                if n_val not in ref_lines_by_n:
                    ref_lines_by_n[n_val] = line_str

        for num in sorted(source_ns):
            cnt = ref_line_counts.get(num, 0)
            if cnt == 0:
                problems.append(f"missing reference line for source [{num}]")
            elif cnt > 1:
                problems.append(f"duplicate reference line for source [{num}] (found {cnt} times)")

        for num in sorted(ref_line_counts.keys()):
            if num not in source_ns:
                problems.append(f"reference line [{num}] has no matching entry in sources.json")

        for num, line_str in sorted(ref_lines_by_n.items()):
            raw_urls = re.findall(r"https?://\S+", line_str)
            urls = [u.rstrip(".,;)") for u in raw_urls]
            if len(urls) != 1:
                problems.append(f"reference line [{num}] must contain exactly one URL (found {len(urls)})")
            elif num in by_n:
                expected_url = by_n[num].get("url", "").rstrip(".,;)")
                if urls[0] != expected_url:
                    problems.append(f"reference line [{num}] URL '{urls[0]}' does not match sources.json URL '{expected_url}'")

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
