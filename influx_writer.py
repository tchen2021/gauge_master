#!/usr/bin/env python3
"""
influx_writer.py - Step 3 of the gauge monitoring system.

Follows readings.jsonl (written by `gauge_reader.py --log readings.jsonl`) and
sends every new reading to InfluxDB, where the Grafana dashboard reads it.
Uses only the Python standard library.

  python influx_writer.py                     # follow readings.jsonl
  python influx_writer.py --from-start        # also send readings already in the file

The defaults match dashboard/docker-compose.yml. If InfluxDB is down, readings
are buffered and sent when it comes back.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

MAX_BUFFER = 20_000  # about 3 hours of readings at 2 per second


def _escape_tag(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace(",", "\\,").replace("=", "\\=").replace(" ", "\\ ")


def _escape_str(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def to_line_protocol(r: dict) -> str:
    """One reading -> one InfluxDB line, e.g.
    gauge,gauge_id=gauge-1,unit=PSI value=62.98,raw_value=62.9,confidence=0.81,ok=1i,status="ok" 1763695042811
    """
    if r.get("type") == "light":
        # indicator light: on=1/0 (missing while the camera can't see it), brightness score
        ts_ms = int(datetime.fromisoformat(r["timestamp"]).timestamp() * 1000)
        fields = [f"score={float(r.get('score') or 0.0)}",
                  f"ok={0 if r.get('status') == 'unavailable' else 1}i"]
        if r.get("on") is not None:
            fields.append(f"on={1 if r['on'] else 0}i")
        return f"light,light_id={_escape_tag(r.get('light_id', 'light'))} {','.join(fields)} {ts_ms}"
    tags = f"gauge,gauge_id={_escape_tag(r.get('gauge_id', 'gauge'))},unit={_escape_tag(r.get('unit') or 'none')}"
    status = r.get("status", "unknown")
    fields = []
    if r.get("value") is not None:
        fields.append(f"value={float(r['value'])}")
    if r.get("raw_value") is not None:
        fields.append(f"raw_value={float(r['raw_value'])}")
    fields.append(f"confidence={float(r.get('confidence') or 0.0)}")
    fields.append(f"ok={0 if status == 'unavailable' else 1}i")
    fields.append(f'status="{_escape_str(status)}"')
    if r.get("reason"):
        fields.append(f'reason="{_escape_str(r["reason"])}"')
    ts_ms = int(datetime.fromisoformat(r["timestamp"]).timestamp() * 1000)
    return f"{tags} {','.join(fields)} {ts_ms}"


class InfluxClient:
    def __init__(self, url: str, org: str, bucket: str, token: str):
        query = urllib.parse.urlencode({"org": org, "bucket": bucket, "precision": "ms"})
        self.write_url = f"{url.rstrip('/')}/api/v2/write?{query}"
        self.headers = {"Authorization": f"Token {token}",
                        "Content-Type": "text/plain; charset=utf-8"}

    def write(self, lines: list[str]) -> None:
        req = urllib.request.Request(self.write_url, data="\n".join(lines).encode(),
                                     headers=self.headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10):
                pass
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            raise RuntimeError(f"InfluxDB returned {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Cannot reach InfluxDB ({e.reason}). "
                               "Is it running? Try: docker compose up -d") from e


class FileFollower:
    """Like `tail -f`: yields complete new lines; handles the file being recreated."""

    def __init__(self, path: Path, from_start: bool):
        self.path, self.from_start = path, from_start
        self.f = None
        self.partial = ""

    def _open(self) -> bool:
        if not self.path.exists():
            return False
        self.f = self.path.open("r")
        if not self.from_start:
            self.f.seek(0, os.SEEK_END)
        self.from_start = True  # a recreated file is read from its start
        return True

    def read_lines(self) -> list[str]:
        if self.f is None and not self._open():
            return []
        # File deleted or truncated (e.g. you cleared the log): reopen it.
        try:
            if self.path.stat().st_size < self.f.tell():
                self.f.close()
                self.f, self.partial = None, ""
                return [] if not self._open() else self.read_lines()
        except FileNotFoundError:
            self.f.close()
            self.f, self.partial = None, ""
            return []
        data = self.partial + self.f.read()
        lines = data.split("\n")
        self.partial = lines.pop()  # last piece may be an incomplete line
        return [ln for ln in lines if ln.strip()]


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S}  {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Send gauge readings to InfluxDB.")
    ap.add_argument("--file", type=Path, default=Path("readings.jsonl"))
    ap.add_argument("--from-start", action="store_true",
                    help="also send readings already in the file")
    ap.add_argument("--url", default=os.environ.get("INFLUX_URL", "http://localhost:8086"))
    ap.add_argument("--org", default=os.environ.get("INFLUX_ORG", "gauge-lab"))
    ap.add_argument("--bucket", default=os.environ.get("INFLUX_BUCKET", "gauges"))
    ap.add_argument("--token", default=os.environ.get("INFLUX_TOKEN", "gauge-dev-token-change-me"))
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between sends")
    args = ap.parse_args()

    client = InfluxClient(args.url, args.org, args.bucket, args.token)
    follower = FileFollower(args.file, args.from_start)
    buffer: list[str] = []
    sent_total, last_report, warned_missing = 0, time.monotonic(), False
    error: Optional[str] = None
    log(f"Following {args.file} -> {args.url} (bucket '{args.bucket}'). Ctrl+C to stop.")

    try:
        while True:
            if follower.f is None and not args.file.exists() and not warned_missing:
                log(f"Waiting for {args.file} - start gauge_reader.py with --log {args.file}")
                warned_missing = True
            for line in follower.read_lines():
                try:
                    buffer.append(to_line_protocol(json.loads(line)))
                except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                    log(f"Skipping bad line: {e}")
            if len(buffer) > MAX_BUFFER:
                dropped = len(buffer) - MAX_BUFFER
                buffer = buffer[dropped:]
                log(f"Buffer full, dropped {dropped} oldest readings")

            while buffer:
                batch = buffer[:5000]
                try:
                    client.write(batch)
                except RuntimeError as e:
                    if str(e) != error:
                        log(f"Write failed, will keep retrying: {e}")
                        error = str(e)
                    break
                if error:
                    log("InfluxDB reachable again.")
                    error = None
                del buffer[:len(batch)]
                sent_total += len(batch)

            if time.monotonic() - last_report >= 10:
                last_report = time.monotonic()
                pending = f", {len(buffer)} waiting" if buffer else ""
                log(f"{sent_total} readings sent{pending}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log(f"Stopped. {sent_total} readings sent.")


if __name__ == "__main__":
    main()
