"""Full-text retrieval: arXiv HTML first, PDF as fallback.

HTML is preferred over PDF for two reasons: LaTeXML keeps every formula as LaTeX source in
the `alttext` attribute (a PDF text layer turns the same formula into unreadable glyph
soup), and the extracted text is roughly half the tokens of the equivalent PDF document
block. HTML exists for LaTeX submissions from late 2023 onwards; older papers and
PDF-only submissions fall through to pypdf.
"""

from __future__ import annotations

import json
import re
from urllib.parse import urljoin

import httpx
from lxml import html as lxml_html

from ..config import CACHE_DIR

USER_AGENT = "paper-news-agent/0.1 (personal daily digest; contact via repo)"

# Below this, an extraction is treated as failed even if it did not raise.
_MIN_USABLE_CHARS = 3_000

# Sections dropped first when the text has to be trimmed to fit a token budget.
_LOW_VALUE = re.compile(
    r"^\s*(\d+(\.\d+)*\.?\s*)?(related work|background and related work|acknowledg(e)?ments?"
    r"|references|bibliography|appendix|supplementary|broader impact"
    r"|ethics statement|reproducibility statement)\b",
    re.I,
)
_DROP_CLASSES = (
    "ltx_bibliography",
    "ltx_appendix",
    "ltx_acknowledgements",
    "ltx_pagination",
    "ltx_tag_section",
)


def fetch(arxiv_id: str, max_chars: int = 90_000, use_cache: bool = True) -> dict:
    """Return `{text, source, chars, truncated, dropped_sections, figures, anchors}`.

    `source` is one of `html`, `pdf`, `none`. `max_chars` ~ 4 chars/token, so the default
    is roughly a 22k-token ceiling on paper text. `figures` carries absolute image URLs so
    the digest can show the paper's own figures (empty on the PDF fallback path).
    """
    cache_path = CACHE_DIR / "fulltext" / f"{arxiv_id}.md"
    figs_path = CACHE_DIR / "fulltext" / f"{arxiv_id}.figures.json"
    if use_cache and cache_path.exists():
        text = cache_path.read_text(encoding="utf-8")
        source = "html" if text.startswith("<!--source:html") else "pdf"
        body = text.split("-->", 1)[1].lstrip() if text.startswith("<!--source:") else text
        figures, anchors = [], {}
        if figs_path.exists():
            side = json.loads(figs_path.read_text(encoding="utf-8"))
            if isinstance(side, dict):
                figures, anchors = side.get("figures", []), side.get("anchors", {})
            else:
                figures = side  # cache written before anchors existed
        return _finish(body, source, max_chars, figures, anchors)

    text, source, figures, anchors = "", "none", [], {}
    try:
        text, figures, anchors = _fetch_html(arxiv_id)
        source = "html"
    except Exception:
        pass
    # The PDF fallback used to fire only on an exception, so a page that parsed cleanly
    # but yielded almost no prose (a stub, a LaTeXML conversion error, a structure the
    # walker does not recognise) silently produced an abstract-only digest even though a
    # perfectly good PDF existed. Judge on what came out, not on whether it threw.
    if len(text) < _MIN_USABLE_CHARS:
        try:
            pdf_text = _fetch_pdf(arxiv_id)
            if len(pdf_text) > len(text):
                # A PDF has no structural ids, so nothing in it is citable evidence.
                text, source, anchors = pdf_text, "pdf", {}
        except Exception:
            pass
    if not text:
        return {
            "text": "",
            "source": "none",
            "chars": 0,
            "truncated": False,
            "dropped_sections": [],
            "figures": [],
        }

    if use_cache and text:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(f"<!--source:{source}-->\n{text}", encoding="utf-8")
        figs_path.write_text(
            json.dumps({"figures": figures, "anchors": anchors}, ensure_ascii=False),
            encoding="utf-8",
        )
    return _finish(text, source, max_chars, figures, anchors)


def _finish(
    text: str, source: str, max_chars: int,
    figures: list | None = None, anchors: dict | None = None,
) -> dict:
    dropped: list[str] = []
    hard_cut = False
    if len(text) > max_chars:
        text, dropped, hard_cut = _trim(text, max_chars)
    return {
        "text": text,
        "source": source,
        "chars": len(text),
        # Dropping related work still leaves a faithful digest; a hard cut does not, so
        # only the latter downgrades the summary's confidence.
        "truncated": hard_cut,
        "dropped_sections": dropped,
        "figures": figures or [],
        # Only ids that survived trimming remain citable.
        "anchors": {
            k: v for k, v in (anchors or {}).items() if _ANCHOR.format(k) in text
        },
    }


def _fetch_html(arxiv_id: str) -> tuple[str, list[dict], dict]:
    # No version suffix: arxiv.org/html/<id> resolves to the latest version.
    url = f"https://arxiv.org/html/{arxiv_id}"
    with httpx.Client(
        timeout=60.0, headers={"User-Agent": USER_AGENT}, follow_redirects=True
    ) as client:
        resp = client.get(url)
        resp.raise_for_status()
        if "text/html" not in resp.headers.get("content-type", ""):
            raise ValueError("not HTML")
        body = resp.text
        base = str(resp.url)
    if len(body) < 2000:
        raise ValueError("HTML too short to be a paper")
    doc = lxml_html.fromstring(body)
    anchors: dict[str, dict] = {}
    # Inline the math *before* reading captions. MathML holds both the presentation glyph
    # and the TeX annotation (`<mi>K</mi>` + `<annotation>K</annotation>`), so a raw
    # `text_content()` on a figcaption yields "length (KK)". Replacing each <math> with
    # its alttext fixes the caption and makes it renderable by KaTeX at the same time.
    _inline_math(doc)
    figures = _extract_figures(doc, base)
    return _markdown_from_doc(doc, anchors), figures, anchors


# Site chrome rather than paper content: arXiv's own logos, and the 1x1 base64 spacers
# LaTeXML emits for layout.
_NOT_A_FIGURE = ("/static/", "data:")


def _extract_figures(doc, base_url: str) -> list[dict]:
    """Pull the paper's own figures out, with absolute image URLs.

    `src` is relative and already carries the version directory
    (`2607.26784v1/x1.png`), so resolving against the final response URL yields
    `https://arxiv.org/html/2607.26784v1/x1.png`.
    """
    out: list[dict] = []
    for fig in doc.xpath("//figure"):
        cls = fig.get("class") or ""
        kind = "table" if "ltx_table" in cls else "figure"
        srcs = [
            urljoin(base_url, src)
            for src in fig.xpath(".//img/@src")
            if src and not src.startswith(_NOT_A_FIGURE)
        ]
        if not srcs:
            continue
        cap_nodes = fig.xpath(".//figcaption")
        caption = _clean(cap_nodes[0].text_content()) if cap_nodes else ""
        label = ""
        number = None
        m = re.match(r"\s*(Figure|Fig\.?|Table)\s*([0-9]+)", caption, re.I)
        if m:
            label = f"{m.group(1).rstrip('.')} {m.group(2)}"
            number = int(m.group(2))
        out.append(
            {
                "kind": kind,
                "number": number,
                "label": label,
                "id": fig.get("id") or "",
                "src": srcs[0],
                "extra_srcs": srcs[1:],
                "caption": caption,
            }
        )
    return out


def _html_to_markdown(source: str) -> str:
    """Parse + normalise + render. Kept as the entry point used by tests."""
    doc = lxml_html.fromstring(source)
    _inline_math(doc)
    return _markdown_from_doc(doc)


def _inline_math(doc) -> None:
    """Replace every <math> subtree with `$alttext$`, in place.

    LaTeXML stores the LaTeX source in @alttext; using it gives the model real notation
    instead of flattened glyphs, and avoids the doubled-symbol problem that reading
    MathML's text content produces.
    """
    for math in doc.xpath("//math"):
        alt = (math.get("alttext") or "").strip()
        if not alt:
            # Not every element carries alttext; the rendered glyphs beat an empty $$,
            # which would otherwise pair up with the next formula's delimiter and wrap
            # a sentence of prose in math markers.
            alt = _clean(math.text_content())
        display = (math.get("display") or "") == "block"
        if not alt:
            replacement = " "
        else:
            replacement = f"\n$$ {alt} $$\n" if display else f" ${alt}$ "
        tail = math.tail or ""
        parent = math.getparent()
        if parent is None:
            continue
        prev = math.getprevious()
        if prev is not None:
            prev.tail = (prev.tail or "") + replacement + tail
        else:
            parent.text = (parent.text or "") + replacement + tail
        parent.remove(math)


# Evidence anchors. LaTeXML emits stable, hierarchical ids — `S2` for a section,
# `S2.SS1` for a subsection, `S2.p3` for a paragraph, `S2.T1` for a table, `S1.F1` for a
# figure. They are what makes "cite the paragraph you got this from" checkable: the
# reviewer stage returns ids, and the backend rejects any that are not in this index.
_ANCHOR = "[[{}]]"


def _markdown_from_doc(doc, anchors: dict | None = None) -> str:
    """Render an already math-inlined document to markdown. Mutates `doc`.

    When `anchors` is passed it is filled with `{id: {kind, section}}` for every block
    that carries one, and each block is prefixed with `[[id]]` in the text.
    """
    for cls in _DROP_CLASSES:
        for node in doc.xpath(f"//*[contains(@class, '{cls}')]"):
            parent = node.getparent()
            if parent is not None:
                parent.remove(node)
    for node in doc.xpath("//script | //style | //nav | //footer"):
        parent = node.getparent()
        if parent is not None:
            parent.remove(node)

    root = doc.xpath("//*[contains(@class,'ltx_page_content')]")
    scope = root[0] if root else doc

    lines: list[str] = []
    _walk(scope, lines, anchors if anchors is not None else {}, {"section": ""})

    text = "\n\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _has_class(node, name: str) -> bool:
    """Exact class-token match.

    A substring test is wrong here and fails loudly in exactly one place: LaTeXML puts
    `ltx_authors_1line` on the document root, so `"ltx_authors" in cls` treats the entire
    paper as an author block — `_walk` emits it as one blob and returns without ever
    descending into the body.
    """
    return name in (node.get("class") or "").split()


def _walk(node, lines: list[str], anchors: dict, state: dict) -> None:
    """Depth-first emit, never descending into a node we have already rendered.

    A flat `iter()` over the tree emits the abstract twice — once for the
    `ltx_abstract` container and again for the `ltx_p` paragraph inside it — which both
    inflates the token count and makes the model treat the repetition as emphasis.
    """
    tag = node.tag if isinstance(node.tag, str) else ""

    if _has_class(node, "ltx_authors"):
        # LaTeXML puts names and affiliations in ltx_authors / ltx_personname, none of
        # which match the paragraph or heading handlers below — so without this branch the
        # whole author block is silently dropped and the model has no way to report
        # institutions. Verified: extracted text for 2607.26784 contained no affiliation
        # string at all before this was added.
        who = _clean(node.text_content())
        if who:
            lines.append("\n## Authors and affiliations\n" + who + "\n")
        return
    if _has_class(node, "ltx_abstract"):
        body = _clean(node.text_content())
        body = re.sub(r"^\s*abstract\s*:?\s*", "", body, flags=re.I)
        body = re.sub(r"^\s*abstract\s*:?\s*", "", body, flags=re.I)  # nested heading
        if body:
            lines.append("\n## Abstract\n" + body + "\n")
        return
    if _has_class(node, "ltx_caption"):
        cap = _clean(node.text_content())
        if cap:
            parent = node.getparent()
            aid = parent.get("id") if parent is not None else None
            kind = "table" if (parent is not None and _has_class(parent, "ltx_table")) else "figure"
            lines.append(_tag(aid, kind, f"[{cap}]", anchors, state))
        return
    if tag == "table":
        rendered = _table(node)
        if rendered:
            holder = node.getparent()
            aid = node.get("id") or (holder.get("id") if holder is not None else None)
            lines.append(_tag(aid, "table", rendered, anchors, state))
        return
    if tag in ("h1", "h2", "h3", "h4", "h5", "h6") or _has_class(node, "ltx_title_section"):
        title = _clean(node.text_content())
        if title:
            level = "##" if tag in ("h1", "h2") else "###"
            holder = node.getparent()
            aid = holder.get("id") if holder is not None else None
            if aid:
                state["section"] = title
            lines.append(_tag(aid, "section", f"\n{level} {title}\n", anchors, state))
        return
    if tag == "p" and _has_class(node, "ltx_p"):
        para = _clean(node.text_content())
        if para:
            # The paragraph id lives on the enclosing ltx_para div, not the <p>.
            holder = node.getparent()
            aid = None
            while holder is not None and aid is None:
                if _has_class(holder, "ltx_para"):
                    aid = holder.get("id")
                    break
                holder = holder.getparent()
            lines.append(_tag(aid, "paragraph", para, anchors, state))
        return

    for child in node:
        _walk(child, lines, anchors, state)


def _tag(aid: str | None, kind: str, body: str, anchors: dict, state: dict) -> str:
    """Prefix a block with its anchor and register it as citable evidence."""
    if not aid:
        return body
    anchors[aid] = {"kind": kind, "section": state.get("section", "")}
    marker = _ANCHOR.format(aid)
    leading = "\n" if body.startswith("\n") else ""
    return f"{leading}{marker} {body.lstrip()}"


def _table(node) -> str:
    rows = []
    for tr in node.xpath(".//tr"):
        cells = [_clean(td.text_content()) for td in tr.xpath("./td | ./th")]
        if any(cells):
            rows.append(" | ".join(cells))
    if not rows:
        return ""
    return "\n".join(rows[:40])  # long result tables add tokens without adding signal


def _clean(text: str) -> str:
    out = re.sub(r"[ \t\xa0]+", " ", (text or "").replace("\n", " "))
    # Inlining math pads with spaces, which leaves `length $K$ , with` in captions.
    out = re.sub(r"\s+([,.;:!?)\]])", r"\1", out)
    return out.strip()


def _fetch_pdf(arxiv_id: str) -> str:
    from io import BytesIO

    from pypdf import PdfReader

    url = f"https://arxiv.org/pdf/{arxiv_id}"
    with httpx.Client(
        timeout=90.0, headers={"User-Agent": USER_AGENT}, follow_redirects=True
    ) as client:
        resp = client.get(url)
        resp.raise_for_status()
        data = resp.content
    reader = PdfReader(BytesIO(data))
    pages = [(page.extract_text() or "") for page in reader.pages[:40]]
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    if len(text) < 1000:
        raise ValueError("PDF text layer is empty or unusable (likely a scan)")
    return text


def _trim(text: str, max_chars: int) -> tuple[str, list[str], bool]:
    """Drop low-value sections before hard-truncating.

    Related work, acknowledgements and appendices go first: they are the least useful
    part of a digest and often a third of the paper. Only if that is not enough do we cut
    the tail, which the caller surfaces as `confidence: low`.
    """
    blocks = re.split(r"(?m)^(?=## )", text)
    kept, dropped = [], []
    for block in blocks:
        heading = block.split("\n", 1)[0].lstrip("# ").strip()
        if _LOW_VALUE.match(heading):
            dropped.append(heading)
        else:
            kept.append(block)
    out = "".join(kept).strip()
    hard_cut = False
    if len(out) > max_chars:
        out = out[:max_chars].rsplit("\n", 1)[0] + "\n\n[...text truncated...]"
        hard_cut = True
    return out, dropped, hard_cut
