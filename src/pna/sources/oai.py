"""arXiv OAI-PMH harvesting.

Notes that cost real debugging time, recorded here so they are not rediscovered:

* The endpoint moved. `export.arxiv.org/oai2` now 301-redirects to
  `https://oaipmh.arxiv.org/oai`; we call the new host directly.
* `from`/`until` filter on the OAI **datestamp** (when the metadata record last changed),
  not on submission date. A day's response therefore mixes genuinely new submissions with
  old papers whose metadata was edited — on a sample weekday, 830 of 922 `cs` records
  were new submissions and the rest were revisions of papers going back to 2021.
  `<updated>` is present on essentially every record, so it cannot be used to tell them
  apart. The workable discriminator is the gap between `<created>` and the datestamp.
* arXiv announces on weekdays only; a weekend `from`/`until` returns `noRecordsMatch`,
  which is a normal empty result and not an error.
* arXiv uses HTTP 503 with `Retry-After` for flow control, not to signal failure.
"""

from __future__ import annotations

import datetime as dt
import time
from typing import Iterator

import httpx
from lxml import etree

OAI_NS = "http://www.openarchives.org/OAI/2.0/"
ARX_NS = "http://arxiv.org/OAI/arXiv/"
DEFAULT_BASE = "https://oaipmh.arxiv.org/oai"
USER_AGENT = "paper-news-agent/0.1 (personal daily digest; contact via repo)"


class OAIError(RuntimeError):
    pass


def _q(tag: str, ns: str = OAI_NS) -> str:
    return f"{{{ns}}}{tag}"


def _text(node, tag: str, ns: str = ARX_NS) -> str | None:
    found = node.find(_q(tag, ns))
    if found is None or found.text is None:
        return None
    return " ".join(found.text.split()) or None


def harvest(
    date_from: str,
    date_until: str,
    sets: list[str],
    base: str = DEFAULT_BASE,
    max_pages: int = 60,
    sleep: float = 3.0,
    timeout: float = 90.0,
) -> Iterator[dict]:
    """Yield raw records for the given datestamp range, deduplicated across sets."""
    seen: set[str] = set()
    with httpx.Client(
        timeout=timeout, headers={"User-Agent": USER_AGENT}, follow_redirects=True
    ) as client:
        for set_spec in sets:
            params = {
                "verb": "ListRecords",
                "metadataPrefix": "arXiv",
                "from": date_from,
                "until": date_until,
                "set": set_spec,
            }
            for page in range(max_pages):
                root = _fetch(client, base, params)
                error = root.find(_q("error"))
                if error is not None:
                    code = error.get("code")
                    if code == "noRecordsMatch":
                        break  # normal: weekend, or nothing in this set
                    raise OAIError(f"OAI error {code}: {error.text}")

                list_records = root.find(_q("ListRecords"))
                if list_records is None:
                    break
                for record in list_records.findall(_q("record")):
                    parsed = _parse_record(record)
                    if parsed and parsed["arxiv_id"] not in seen:
                        seen.add(parsed["arxiv_id"])
                        yield parsed

                token_node = list_records.find(_q("resumptionToken"))
                token = (token_node.text or "").strip() if token_node is not None else ""
                if not token:
                    break
                # A resumptionToken replaces every other argument.
                params = {"verb": "ListRecords", "resumptionToken": token}
                time.sleep(sleep)
            else:
                raise OAIError(
                    f"set={set_spec} still paginating after {max_pages} pages; "
                    f"narrow the date range or raise max_pages."
                )


# Transport-level failures that say nothing about the request and everything about the
# network between here and arXiv. A full harvest is several hundred requests over a
# quarter of an hour, so hitting one of these is close to certain rather than unlucky:
# a scheduled run died on `[Errno 54] Connection reset by peer` after 16 minutes of
# successful pagination. Retrying is the whole fix; the previous code retried 503 flow
# control diligently and then let a TCP reset out of the loop untouched.
_TRANSIENT = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.ConnectTimeout,
    httpx.RemoteProtocolError,
    httpx.PoolTimeout,
)


def _fetch(client: httpx.Client, base: str, params: dict, attempts: int = 6):
    delay = 5.0
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = client.get(base, params=params)
        except _TRANSIENT as exc:
            last = exc
            if attempt == attempts - 1:
                break
            time.sleep(min(delay, 120.0))
            delay = min(delay * 2, 120.0)
            continue
        if resp.status_code == 503:
            wait = float(resp.headers.get("Retry-After", delay) or delay)
            time.sleep(min(wait, 120.0))
            delay = min(delay * 2, 120.0)
            continue
        if resp.status_code >= 500:
            # A 5xx is the server having a moment, not a bad request. Same treatment.
            last = OAIError(f"OAI returned {resp.status_code}")
            if attempt == attempts - 1:
                break
            time.sleep(min(delay, 120.0))
            delay = min(delay * 2, 120.0)
            continue
        resp.raise_for_status()
        try:
            return etree.fromstring(resp.content)
        except etree.XMLSyntaxError as exc:
            if attempt == attempts - 1:
                raise OAIError(f"unparseable OAI response: {exc}") from exc
            time.sleep(delay)
    if last is not None:
        raise OAIError(
            f"OAI unreachable after {attempts} attempts: {type(last).__name__}: {last}"
        ) from last
    raise OAIError(f"OAI still returning 503 after {attempts} attempts")


def _parse_record(record) -> dict | None:
    header = record.find(_q("header"))
    if header is None:
        return None
    if (header.get("status") or "") == "deleted":
        return None
    meta = record.find(f"{_q('metadata')}/{_q('arXiv', ARX_NS)}")
    if meta is None:
        return None

    arxiv_id = _text(meta, "id")
    title = _text(meta, "title")
    abstract = _text(meta, "abstract")
    if not (arxiv_id and title and abstract):
        return None

    authors = []
    holder = meta.find(_q("authors", ARX_NS))
    if holder is not None:
        for author in holder.findall(_q("author", ARX_NS)):
            fore = _text(author, "forenames") or ""
            key = _text(author, "keyname") or ""
            name = f"{fore} {key}".strip()
            if name:
                authors.append(name)

    categories = (_text(meta, "categories") or "").split()
    datestamp = _text(header, "datestamp", OAI_NS)

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "categories": categories,
        "primary_category": categories[0] if categories else None,
        "created": _text(meta, "created"),
        "updated": _text(meta, "updated"),
        "datestamp": datestamp,
        "comments": _text(meta, "comments"),
        "journal_ref": _text(meta, "journal-ref"),
        "doi": _text(meta, "doi"),
        "license": _text(meta, "license"),
        "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
    }


def is_new_submission(rec: dict, window_days: int) -> bool:
    """True when the record looks like a first announcement rather than a metadata edit.

    See the module docstring: `<updated>` is always populated, so the gap between
    `<created>` and the OAI datestamp is what separates the two.
    """
    created, stamp = rec.get("created"), rec.get("datestamp")
    if not created or not stamp:
        return True  # can't tell; let the later stages judge on content
    try:
        gap = (dt.date.fromisoformat(stamp) - dt.date.fromisoformat(created)).days
    except ValueError:
        return True
    return 0 <= gap <= window_days
