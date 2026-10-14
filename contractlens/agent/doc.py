"""Contract text loading, chunking, search and quote-verification helpers.

Everything here is deterministic and provider-independent. The LLM never sees the
whole contract at once by default: it gets an outline and uses tools to read and
search chunks, which keeps token use (and free-tier rate-limit pressure) low.
"""
import re
from datetime import date
from pathlib import Path

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
_MON = {m.lower(): i + 1 for i, m in enumerate(MONTHS)}
_MON.update({m[:3].lower(): i + 1 for i, m in enumerate(MONTHS)})
_MON_RE = "|".join(sorted(_MON, key=len, reverse=True))

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
         "fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def num_words(n: int) -> str:
    """Spell out 0 <= n < 10,000 ('one hundred five', 'sixty-five')."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + ("-" + _ONES[o] if o else "")
    if n < 1000:
        h, r = divmod(n, 100)
        return _ONES[h] + " hundred" + (" " + num_words(r) if r else "")
    th, r = divmod(n, 1000)
    return _ONES[th] + " thousand" + (" " + num_words(r) if r else "")


def collapse(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def norm(s: str) -> str:
    """Aggressive normalisation for quote matching: case, spacing, hyphens, punctuation."""
    return re.sub(r"[^a-z0-9$%]+", "", s.lower())


# ---------------------------------------------------------------- dates / numbers
_DATE_PATTERNS = [
    (re.compile(rf"\b({_MON_RE})\.? (\d{{1,2}}),? (\d{{4}})\b", re.I), ("m", "d", "y")),
    (re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)? day of ({_MON_RE}),? (\d{{4}})\b", re.I), ("d", "m", "y")),
    (re.compile(rf"\b(\d{{1,2}}) ({_MON_RE})\.? (\d{{4}})\b", re.I), ("d", "m", "y")),
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), ("y", "m", "d")),
]


def date_spans(text: str):
    """[(start_pos, iso_date)] for every date found in collapsed text, in order."""
    t = collapse(text)
    found = []
    for rx, order in _DATE_PATTERNS:
        for m in rx.finditer(t):
            p = dict(zip(order, m.groups()))
            mon = int(p["m"]) if p["m"].isdigit() else _MON[p["m"].lower()]
            try:
                found.append((m.start(), date(int(p["y"]), mon, int(p["d"])).isoformat()))
            except ValueError:
                pass
    return sorted(found)


def dates_in(text: str) -> set:
    return {iso for _, iso in date_spans(text)}


def numbers_in(text: str) -> set:
    return {int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", text) if x.replace(",", "")}


def number_supported(value: int, quote: str) -> bool:
    """True if `value` appears in the quote as digits or as spelled-out words."""
    if value in numbers_in(quote):
        return True
    if 0 <= value < 10000:
        spelled = num_words(value).replace("-", " ")
        return spelled in collapse(quote.lower().replace("-", " "))
    return False


# ---------------------------------------------------------------- chunking
_HEADING = re.compile(r"^(ARTICLE|SECTION|EXHIBIT|AMENDMENT|SCHEDULE|\d+\.\s+[A-Z]{3,})")


def chunk_text(text: str, soft_cap: int = 700, hard_cap: int = 1500):
    """Split into chunks at headings / paragraph breaks, keeping chunks readable."""
    chunks, cur = [], []

    def flush():
        body = "\n".join(cur).strip()
        if body:
            chunks.append(body)
        cur.clear()

    for raw in text.splitlines():
        line = raw.strip()
        size = sum(len(x) + 1 for x in cur)
        if not line:
            if size >= 300:
                flush()
            continue
        is_heading = bool(_HEADING.match(line)) or (line.isupper() and len(line) > 6)
        if is_heading and cur:
            flush()
        elif size >= hard_cap or (size >= soft_cap and line[0].isupper() and cur and cur[-1].endswith(".")):
            flush()
        cur.append(line)
    flush()
    return [{"id": i, "text": c} for i, c in enumerate(chunks)]


class ContractDoc:
    def __init__(self, text: str, name: str = "contract"):
        self.name = name
        self.text = text
        self.chunks = chunk_text(text)
        self._norm_text = norm(text)

    def outline(self, width: int = 70) -> str:
        return "\n".join(f"[{c['id']}] {collapse(c['text'])[:width]}" for c in self.chunks)

    def get(self, chunk_id: int):
        return next((c for c in self.chunks if c["id"] == chunk_id), None)

    def search(self, query: str, k: int = 4):
        tokens = {t for t in re.findall(r"[a-z0-9$%]+", query.lower()) if len(t) > 2 or t.isdigit()}
        scored = []
        for c in self.chunks:
            low = c["text"].lower()
            s = sum(1 for t in tokens if t in low)
            if s:
                scored.append((-s, c["id"], c))
        scored.sort(key=lambda x: (x[0], x[1]))
        return [c for _, _, c in scored[:k]]

    def contains_quote(self, quote: str) -> bool:
        n = norm(quote)
        return len(n) >= 8 and n in self._norm_text


def load_contract(path) -> ContractDoc:
    """Load a .pdf (via pypdf) or .txt contract."""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        text = "\n".join((pg.extract_text() or "") for pg in PdfReader(str(p)).pages)
    else:
        text = p.read_text(encoding="utf-8")
    return ContractDoc(text, name=p.stem)
