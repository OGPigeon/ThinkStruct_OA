"""Text normalization helpers shared by ingestion and query time."""
import re

# Tokens: lowercase alphanumeric runs, so hyphenated compounds split ("tire-pressure" -> tire, pressure)
# and a keyword filter on "tire" still matches. Single characters and purely numeric tokens (reference
# numerals like "100", "12a") are dropped: in patents they are almost always noise.
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NUMERIC_RE = re.compile(r"^\d+[a-z]?$")
MAX_TOKEN_LEN = 30

# English stopwords plus patent boilerplate that appears in nearly every claim.
STOPWORDS = frozenset("""
a an the and or of to in on at by for from with as is are was were be been being it its this that these those
which who whom whose what when where how than then there their them they such so not no nor but if into onto
over under about above below between through during each any all both either neither other another same some
may can could would should will shall must has have had having do does did done also only more most less least
one two first second third said wherein whereby thereof therein thereto comprising comprises comprise comprised
including includes include included consisting consists having claim claims according embodiment embodiments
method system apparatus device plurality least further configured based example examples present disclosure
invention used using use via within without upon
""".split())

# "2 . The ...", "1 - 15 . (canceled)", "1 .- 15 . (canceled)"; the trailing '.' is required so a first
# claim that happens to start with a digit ("3D printed ...") is not mistaken for a number.
_CLAIM_NUM_RE = re.compile(r"^\s*(\d+)\s*\.?\s*(?:-\s*(\d+)\s*)?\.\s*-?\s*")
_CANCELED_RE = re.compile(r"\((?:canceled|cancelled|deleted)\)", re.I)
_DEPENDENT_RE = re.compile(r"\bclaims?\s+\d+|\b(?:any|one) of the (?:preceding|previous|above) claims?", re.I)
_SPACE_PUNCT_RE = re.compile(r"\s+([,.;:)])")
_GLUED_NUMERAL_RE = re.compile(r"(?<=[a-z])(\d+)(?=[a-zA-Z])")
_WS_RE = re.compile(r"\s+")


def tokenize(text):
    """Lowercase word tokens with stopwords and pure numerals removed."""
    out = []
    for t in _TOKEN_RE.findall(text.lower()):
        if len(t) < 2 or len(t) > MAX_TOKEN_LEN or t in STOPWORDS or _NUMERIC_RE.match(t):
            continue
        out.append(t)
    return out


def tidy(text):
    """Collapse whitespace and fix the ' ,' / ' .' spacing artifacts present in the USPTO dumps."""
    return _SPACE_PUNCT_RE.sub(r"\1", _WS_RE.sub(" ", text)).strip()


def unglue_numerals(text):
    """'insert100according' -> 'insert 100 according' (reference numerals glued to words in descriptions)."""
    return _GLUED_NUMERAL_RE.sub(r" \1 ", text)


def parse_claims(raw_claims):
    """Return [(claim_number, text, is_independent)] with canceled claims removed.

    The raw claims carry their own numbering prefix ("2 . The insert of claim 1 , wherein ..."), except
    the first one. Canceled claims can cover ranges ("1 - 15 . (canceled)"), so numbering is parsed
    rather than inferred from list position.
    """
    out, next_num = [], 1
    for raw in raw_claims or []:
        if not isinstance(raw, str) or not raw.strip():
            continue
        m = _CLAIM_NUM_RE.match(raw)
        if m:
            num = int(m.group(1))
            last = int(m.group(2)) if m.group(2) else num
            body = raw[m.end():]
        else:
            num = last = next_num
            body = raw
        next_num = last + 1
        if _CANCELED_RE.search(body) and len(body) < 40:
            continue
        body = tidy(body)
        if body:
            out.append((num, body, not _DEPENDENT_RE.search(body)))
    return out


def normalize_classification(code):
    """Search key for a classification code: uppercase, no spaces, without the small dataset's 'FI' suffix.

    The small dataset stores codes as 'B60B304FI' and the large one as 'H01L21/0234'; both keep their
    subclass (first 4 chars) so prefix filters like 'B60B' work across both. 'N/A' / empty -> None.
    """
    if not code or not isinstance(code, str):
        return None
    key = code.upper().replace(" ", "")
    if key in ("N/A", "NA", "NONE"):
        return None
    if key.endswith("FI") and "/" not in key:
        key = key[:-2]
    return key or None
