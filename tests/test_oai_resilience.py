"""Transient network failure during a harvest.

A full harvest is several hundred paginated requests over ~15 minutes. A TCP reset
somewhere in there is close to certain rather than unlucky — one killed a scheduled run
after 16 minutes of successful pagination and discarded every record it had collected.
"""

import httpx
import pytest

_REQ = httpx.Request("GET", "http://x")

from pna.sources import oai


class _Client:
    """Fails the first `fail_times` calls, then serves `body`."""

    def __init__(self, fail_times, body, exc=None):
        self.fail_times = fail_times
        self.body = body
        self.calls = 0
        self.exc = exc or httpx.ConnectError("[Errno 54] Connection reset by peer")

    def get(self, base, params=None):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc
        return httpx.Response(200, content=self.body, request=_REQ)


_EMPTY = (
    b'<?xml version="1.0"?><OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">'
    b'<error code="noRecordsMatch">nothing</error></OAI-PMH>'
)


def test_a_connection_reset_is_retried_not_fatal(monkeypatch):
    monkeypatch.setattr(oai.time, "sleep", lambda _s: None)
    client = _Client(fail_times=2, body=_EMPTY)
    root = oai._fetch(client, "http://x", {})
    assert root is not None
    assert client.calls == 3, "should have retried twice before succeeding"


@pytest.mark.parametrize("exc", [
    httpx.ConnectError("reset"),
    httpx.ReadTimeout("slow"),
    httpx.RemoteProtocolError("truncated"),
    httpx.PoolTimeout("busy"),
])
def test_every_transport_failure_class_is_retried(monkeypatch, exc):
    """The original code retried 503 flow control diligently and let these straight out."""
    monkeypatch.setattr(oai.time, "sleep", lambda _s: None)
    client = _Client(fail_times=1, body=_EMPTY, exc=exc)
    assert oai._fetch(client, "http://x", {}) is not None


def test_giving_up_names_the_underlying_cause(monkeypatch):
    """A bare 'still returning 503' after a TCP reset sent debugging the wrong way."""
    monkeypatch.setattr(oai.time, "sleep", lambda _s: None)
    client = _Client(fail_times=99, body=_EMPTY)
    with pytest.raises(oai.OAIError) as err:
        oai._fetch(client, "http://x", {}, attempts=3)
    assert "ConnectError" in str(err.value)
    assert client.calls == 3


def test_a_5xx_is_retried_rather_than_raised(monkeypatch):
    monkeypatch.setattr(oai.time, "sleep", lambda _s: None)

    class Flaky:
        calls = 0

        def get(self, base, params=None):
            Flaky.calls += 1
            if Flaky.calls == 1:
                return httpx.Response(502, content=b"bad gateway", request=_REQ)
            return httpx.Response(200, content=_EMPTY, request=_REQ)

    assert oai._fetch(Flaky(), "http://x", {}) is not None
    assert Flaky.calls == 2


def test_a_failed_harvest_keeps_what_it_already_collected(tmp_path, monkeypatch):
    """The 16 minutes of successful pagination before the reset must not be thrown away.

    `merge_day` is keyed on `arxiv_id`, so keeping partial work costs nothing and a re-run
    tops it up. The command still reports failure: an incomplete day that went on to be
    scored and published would be marked done, which is worse than no page at all.
    """
    import argparse

    from pna import cli
    from pna.config import load_config

    saved: dict = {}
    monkeypatch.setattr(cli, "merge_day",
                        lambda date, recs: (saved.setdefault(date, recs), (len(recs), 0))[1])
    monkeypatch.setattr(cli, "write_run_log", lambda *a, **k: None)

    def flaky_harvest(*a, **k):
        for i in range(3):
            yield {"arxiv_id": f"2608.{i}", "title": "T", "abstract": "A",
                   "created": "2026-08-20", "datestamp": "2026-08-20", "categories": ["cs.LG"]}
        raise oai.OAIError("OAI unreachable after 6 attempts: ConnectError: reset")

    monkeypatch.setattr(cli.oai, "harvest", flaky_harvest)
    args = argparse.Namespace(date="2026-08-20", date_from=None, include_revisions=True)
    code = cli.cmd_ingest(args, load_config())

    assert code == 1, "an incomplete day must not let the pipeline continue"
    assert len(saved["2026-08-20"]) == 3, "records collected before the failure are kept"
