"""Minimal Google Patents scraper so any US publication number can be used as a query, even if it is not
in the index. Returns the same shape as PatentIndex.get_patent()."""
import html
import re
import urllib.request
from html.parser import HTMLParser

from .text import parse_claims, tidy


class _Parser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title, self.abstract, self.codes, self.claims = "", "", [], []
        self._stack = []        # (tag, role) for every open element, so we know when a section closes
        self._buf = None
        self._role = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        role = None
        cls = a.get("class") or ""
        if tag == "div" and cls.split() and cls.split()[0] in ("claim", "claim-dependent") and a.get("num"):
            role = "claim"
        elif tag == "div" and cls == "abstract":
            role = "abstract"
        elif tag == "span" and a.get("itemprop") == "title" and not self.title:
            role = "title"
        elif tag == "span" and a.get("itemprop") == "Code":
            role = "code"
        if tag in ("br", "img", "meta", "link", "input", "hr"):
            return
        if role and self._role is None:
            self._role, self._buf = role, []
            self._stack.append((tag, role))
        else:
            self._stack.append((tag, None))

    def handle_endtag(self, tag):
        while self._stack:
            t, role = self._stack.pop()
            if role:
                text = tidy(html.unescape("".join(self._buf)))
                if role == "claim":
                    self.claims.append(text)
                elif role == "abstract":
                    self.abstract = text
                elif role == "title":
                    self.title = text
                elif role == "code":
                    self.codes.append(text)
                self._role, self._buf = None, None
            if t == tag:
                break

    def handle_data(self, data):
        if self._buf is not None:
            self._buf.append(data)


def fetch_patent(number, timeout=20):
    num = re.sub(r"\s", "", str(number)).upper()
    if not num.startswith("US"):
        num = "US" + num
    if not re.search(r"[AB]\d$", num):
        num += "A1"
    req = urllib.request.Request(f"https://patents.google.com/patent/{num}/en", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        page = resp.read().decode("utf-8", "replace")
    p = _Parser()
    p.feed(page)
    if not (p.claims or p.abstract):
        raise ValueError(f"could not find claims or abstract for {num}")
    # Google prefixes each claim with its number ("1. A system ..."); parse_claims handles that format.
    claims = parse_claims([re.sub(r"^(\d+)\s*\.", r"\1 .", c) for c in p.claims])
    leaf_codes = [c for c in p.codes if "/" in c]
    return {"id": None, "doc_number": re.sub(r"\D", "", num[2:])[:11], "title": p.title, "abstract": p.abstract,
            "classification": leaf_codes[0] if leaf_codes else None, "date": None, "claims": claims, "remote": True}
