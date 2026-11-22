#!/usr/bin/env python3
"""
alert_monitor.py - Step 2 of the gauge monitoring system.

Reads the JSON readings printed by gauge_reader.py and opens a Jira ticket when
a gauge stays outside its limit. Run both together with a pipe:

  python gauge_reader.py --source 1 --log readings.jsonl | python alert_monitor.py --high 30

Alert rules:
  * HOLD:  the value must stay past the limit for --hold seconds (default 5),
           so one bad frame or a needle wobble never creates a ticket.
  * DEDUPE: one ticket per incident. No new ticket until the value recovers.
  * CLEAR: the incident ends only when the value is back inside the limit by
           --clear-margin (hysteresis), so it can't flicker at the boundary.
           A comment with the peak value and duration is added to the ticket.
  * CAMERA: if readings are "unavailable" for --camera-timeout seconds
           (blocked view, glare, bumped camera), a separate ticket is opened.

Jira settings come from environment variables:
  JIRA_URL          e.g. https://your-team.atlassian.net
  JIRA_EMAIL        the Atlassian account email
  JIRA_API_TOKEN    create at https://id.atlassian.com/manage-profile/security/api-tokens
  JIRA_PROJECT_KEY  e.g. OPS
  JIRA_ISSUE_TYPE   optional, default "Task"
Without them (or with --dry-run) tickets are printed instead of created.

Every ticket event (opened / failed / resolved) is also written to InfluxDB
(measurement "incident") so the Grafana dashboard can list the Jira tickets.
The ticket's real Jira status (To Do / In Progress / Done ...) is checked every
--jira-poll seconds and shown on the dashboard too. Tickets are remembered in
tickets.json for 7 days. Defaults match dashboard/docker-compose.yml; use
--no-influx to turn this off.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Protocol

RETRY_SECONDS = 10.0
UNSENT_PATH = Path("unsent_tickets.jsonl")


def log(msg: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"{stamp}  {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Ticketing backends
# --------------------------------------------------------------------------- #
class Ticketer(Protocol):
    def create(self, summary: str, lines: list[str]) -> str: ...
    def comment(self, key: str, lines: list[str]) -> None: ...
    def link(self, key: str) -> str: ...
    def status(self, key: str) -> Optional[tuple[str, str]]: ...


def _adf(lines: list[str]) -> dict:
    """Jira Cloud REST v3 wants descriptions in Atlassian Document Format."""
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": line}]}
            for line in lines if line
        ],
    }


class JiraTicketer:
    def __init__(self, base_url: str, email: str, token: str, project: str,
                 issue_type: str = "Task", labels: Optional[list[str]] = None):
        self.base_url = base_url.rstrip("/")
        self.project, self.issue_type = project, issue_type
        self.labels = labels or ["gauge-monitor"]
        creds = base64.b64encode(f"{email}:{token}".encode()).decode()
        self.headers = {"Authorization": f"Basic {creds}",
                        "Content-Type": "application/json",
                        "Accept": "application/json"}

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.base_url + path, data=json.dumps(body).encode(),
                                     headers=self.headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                text = resp.read().decode()
                return json.loads(text) if text else {}
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            raise RuntimeError(f"Jira returned {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Could not reach Jira: {e.reason}") from e

    def create(self, summary: str, lines: list[str]) -> str:
        fields = {"project": {"key": self.project},
                  "issuetype": {"name": self.issue_type},
                  "summary": summary[:250],
                  "description": _adf(lines),
                  "labels": self.labels}
        return self._post("/rest/api/3/issue", {"fields": fields})["key"]

    def comment(self, key: str, lines: list[str]) -> None:
        self._post(f"/rest/api/3/issue/{key}/comment", {"body": _adf(lines)})

    def link(self, key: str) -> str:
        return f"{self.base_url}/browse/{key}"

    def status(self, key: str) -> Optional[tuple[str, str]]:
        """(status name, category) e.g. ("Done", "done"); category is new/indeterminate/done."""
        req = urllib.request.Request(f"{self.base_url}/rest/api/3/issue/{key}?fields=status",
                                     headers=self.headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                st = json.loads(resp.read().decode())["fields"]["status"]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return ("Deleted", "deleted")
            raise RuntimeError(f"Jira returned {e.code} for {key}") from e
        except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
            raise RuntimeError(f"Could not read status of {key}: {e}") from e
        return st.get("name", "?"), st.get("statusCategory", {}).get("key", "")


class DryRunTicketer:
    def __init__(self) -> None:
        self.count = 0

    def create(self, summary: str, lines: list[str]) -> str:
        self.count += 1
        key = f"DRYRUN-{self.count}"
        log(f"[dry run] would create ticket {key}: {summary}")
        for line in lines:
            print(f"            {line}")
        return key

    def comment(self, key: str, lines: list[str]) -> None:
        log(f"[dry run] would comment on {key}:")
        for line in lines:
            print(f"            {line}")

    def link(self, key: str) -> str:
        return ""

    def status(self, key: str) -> Optional[tuple[str, str]]:
        return None


# --------------------------------------------------------------------------- #
# Ticket history for the Grafana dashboard
# --------------------------------------------------------------------------- #
def _lp_tag(v: str) -> str:
    return str(v).replace("\\", "\\\\").replace(",", "\\,").replace("=", "\\=").replace(" ", "\\ ")


def _lp_str(v: str) -> str:
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'


class IncidentLog:
    """Writes ticket events to InfluxDB. Never stops the monitor if InfluxDB is down."""

    def __init__(self, url: str, org: str, bucket: str, token: str):
        query = urllib.parse.urlencode({"org": org, "bucket": bucket, "precision": "ms"})
        self.write_url = f"{url.rstrip('/')}/api/v2/write?{query}"
        self.headers = {"Authorization": f"Token {token}",
                        "Content-Type": "text/plain; charset=utf-8"}
        self.last_error: Optional[str] = None

    def record(self, gauge_id: str, kind: str, ts: float, fields: dict) -> None:
        parts = []
        for k, v in fields.items():
            if v is None:
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                parts.append(f"{k}={float(v)}")
            else:
                parts.append(f"{k}={_lp_str(v)}")
        line = f"incident,gauge_id={_lp_tag(gauge_id)},kind={_lp_tag(kind)} {','.join(parts)} {int(ts * 1000)}"
        req = urllib.request.Request(self.write_url, data=line.encode(),
                                     headers=self.headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=3):
                pass
            if self.last_error:
                log("Dashboard ticket history: InfluxDB reachable again.")
            self.last_error = None
        except (urllib.error.URLError, OSError) as e:
            msg = str(getattr(e, "reason", e))
            if msg != self.last_error:   # warn once, not on every event
                log(f"Could not record ticket on dashboard (InfluxDB): {msg}")
            self.last_error = msg


class TicketTracker:
    """Remembers the tickets this monitor created (tickets.json) and publishes
    every change - incident state or Jira status - to InfluxDB for the dashboard."""

    KEEP_SECONDS = 7 * 24 * 3600

    def __init__(self, path: Path, incident_log: IncidentLog):
        self.path, self.incident_log = path, incident_log
        self.lock = threading.Lock()
        self.records: dict[str, dict] = {}
        if path.exists():
            try:
                self.records = json.loads(path.read_text())
            except ValueError:
                log(f"Ignoring unreadable {path}")

    def update(self, rec: dict, ts: float) -> None:
        with self.lock:
            key = rec.get("ticket") or ""
            merged = {**self.records.get(key, {}), **{k: v for k, v in rec.items() if v is not None}}
            merged["updated_ts"] = ts
            if key:
                self.records[key] = merged
                cutoff = time.time() - self.KEEP_SECONDS
                self.records = {k: v for k, v in self.records.items()
                                if v.get("updated_ts", 0) >= cutoff}
                self.path.write_text(json.dumps(self.records, indent=2))
            fields = {k: merged.get(k) for k in ("ticket", "state", "pressure", "url", "started", "peak",
                                                 "value", "duration_s", "jira_status",
                                                 "jira_category")}
            fields["ticket"] = fields["ticket"] or ""
            self.incident_log.record(merged["gauge_id"], merged["kind"], ts, fields)

    def tickets(self) -> list[str]:
        with self.lock:
            return list(self.records)


def poll_jira_status(tracker: TicketTracker, ticketer: "Ticketer", every: float) -> None:
    """Background thread: copy each ticket's current Jira status to the dashboard."""
    last_error = None
    while True:
        for key in tracker.tickets():
            try:
                st = ticketer.status(key)
            except RuntimeError as e:
                if str(e) != last_error:
                    log(f"Jira status check failed: {e}")
                last_error = str(e)
                continue
            last_error = None
            if st is None:
                continue
            name, cat = st
            rec = tracker.records.get(key, {})
            if (rec.get("jira_status"), rec.get("jira_category")) != (name, cat):
                log(f"Jira status of {key}: {name}")
                tracker.update({**rec, "jira_status": name, "jira_category": cat}, time.time())
        time.sleep(every)


# --------------------------------------------------------------------------- #
# Alert logic
# --------------------------------------------------------------------------- #
@dataclass
class Incident:
    gauge_id: str
    kind: str                    # high | low | camera
    started: str                 # ISO timestamp of the first bad reading
    started_ts: float
    peak: Optional[float]
    unit: str
    ticket: Optional[str] = None
    last_attempt: float = 0.0


def _ts(reading: dict) -> float:
    return datetime.fromisoformat(reading["timestamp"]).timestamp()


class AlertMonitor:
    def __init__(self, ticketer: Ticketer, high: Optional[float], low: Optional[float],
                 hold: float, clear_margin: float, camera_timeout: float,
                 state_path: Optional[Path] = None,
                 tracker: Optional[TicketTracker] = None):
        self.ticketer = ticketer
        self.tracker = tracker
        self.high, self.low = high, low
        self.hold, self.clear_margin = hold, clear_margin
        self.camera_timeout = camera_timeout
        self.state_path = state_path
        self.pending: dict[str, list] = {}   # key -> [since ts, since iso, peak]
        self.active: dict[str, Incident] = {}
        self._load_state()

    # -- persistence: an open incident survives a restart, so no duplicate tickets
    def _load_state(self) -> None:
        if self.state_path and self.state_path.exists():
            data = json.loads(self.state_path.read_text())
            self.active = {k: Incident(**v) for k, v in data.items()}
            for k, inc in self.active.items():
                log(f"Resuming open incident {k} (ticket {inc.ticket})")

    def _save_state(self) -> None:
        if self.state_path:
            self.state_path.write_text(
                json.dumps({k: asdict(v) for k, v in self.active.items()}, indent=2))

    # -- main entry point
    def process(self, r: dict) -> None:
        now = _ts(r)
        gid = r.get("gauge_id", "gauge")
        value, status = r.get("value"), r.get("status")

        if status == "unavailable" or value is None:
            # Can't confirm a sustained out-of-limit condition without readings.
            self.pending.pop(f"{gid}:high", None)
            self.pending.pop(f"{gid}:low", None)
            self._check(gid, "camera", True, now, r, None)
        else:
            self._check(gid, "camera", False, now, r, None)
            if self.high is not None:
                bad = value >= self.high or status == "above_scale"
                clear = value < self.high - self.clear_margin
                self._check(gid, "high", bad, now, r, value, clear)
            if self.low is not None:
                bad = value <= self.low or status == "below_scale"
                clear = value > self.low + self.clear_margin
                self._check(gid, "low", bad, now, r, value, clear)

        # Retry tickets that failed to be created (e.g. network down).
        for inc in self.active.values():
            if inc.ticket is None and now - inc.last_attempt >= RETRY_SECONDS:
                self._open_ticket(inc, r, now)

    def _check(self, gid: str, kind: str, bad: bool, now: float, r: dict,
               value: Optional[float], clear: Optional[bool] = None) -> None:
        key = f"{gid}:{kind}"
        if clear is None:
            clear = not bad
        hold = self.camera_timeout if kind == "camera" else self.hold

        if bad:
            pend = self.pending.setdefault(key, [now, r["timestamp"], value])
            if value is not None:
                pend[2] = self._worse(kind, pend[2], value)
            inc = self.active.get(key)
            if inc is not None:
                if value is not None:
                    inc.peak = self._worse(kind, inc.peak, value)
                return
            since, since_iso, peak = pend
            if now - since >= hold:
                inc = Incident(gid, kind, since_iso, since, peak, r.get("unit", ""))
                self.active[key] = inc
                self._open_ticket(inc, r, now)
            return

        self.pending.pop(key, None)
        inc = self.active.get(key)
        if inc is not None and clear:
            self._resolve(key, inc, r, now)

    @staticmethod
    def _worse(kind: str, a: Optional[float], b: float) -> float:
        if a is None:
            return b
        return max(a, b) if kind == "high" else min(a, b)

    def _limit_text(self, kind: str) -> str:
        if kind == "high":
            return f"high limit {self.high:g}"
        if kind == "low":
            return f"low limit {self.low:g}"
        return "camera reading"

    @staticmethod
    def _pressure_label(kind: str, state: str) -> str:
        active = state != "resolved"
        if kind == "camera":
            return "Camera Fault" if active else "Camera OK"
        if kind == "low":
            return "Pressure Low" if active else "Pressure High"
        return "Pressure High" if active else "Pressure Low"

    def _record(self, inc: Incident, state: str, now: float, value=None) -> None:
        if self.tracker is None:
            return
        self.tracker.update({
            "gauge_id": inc.gauge_id, "kind": inc.kind,
            "ticket": inc.ticket or "",
            "state": state,                       # open | ticket_failed | resolved
            "pressure": self._pressure_label(inc.kind, state),
            # a brand-new ticket is open; the Jira poller updates this afterwards
            "jira_category": "new" if state == "open" else None,
            "url": self.ticketer.link(inc.ticket) if inc.ticket else "",
            "started": inc.started,
            "peak": inc.peak,
            "value": value,
            "duration_s": round(now - inc.started_ts, 1),
        }, now)

    def _open_ticket(self, inc: Incident, r: dict, now: float) -> None:
        inc.last_attempt = now
        unit = inc.unit
        if inc.kind == "camera":
            summary = f"[{inc.gauge_id}] Gauge camera cannot read the gauge"
            lines = [f"Gauge: {inc.gauge_id}",
                     f"No valid reading since {inc.started} "
                     f"({now - inc.started_ts:.0f} s).",
                     f"Reason reported by the reader: {r.get('reason') or 'unknown'}.",
                     "Check for a blocked view, glare, poor lighting, or a moved camera.",
                     "The pressure itself is NOT being monitored until this is fixed."]
        else:
            word = "HIGH" if inc.kind == "high" else "LOW"
            summary = (f"[{inc.gauge_id}] Pressure {word}: peak {inc.peak:g} {unit} "
                       f"({self._limit_text(inc.kind)} {unit})")
            lines = [f"Gauge: {inc.gauge_id}",
                     f"Peak reading so far: {inc.peak:g} {unit} ({self._limit_text(inc.kind)} {unit}).",
                     f"Out of limit since {inc.started}, held for "
                     f"{now - inc.started_ts:.0f} s.",
                     f"Current reading: {r.get('value')} {unit}, "
                     f"reader confidence {r.get('confidence')}."]
        lines.append("Created automatically by alert_monitor.py.")
        log(f"ALERT {inc.gauge_id}:{inc.kind} -> opening ticket")
        try:
            inc.ticket = self.ticketer.create(summary, lines)
            log(f"Ticket {inc.ticket} created")
            self._record(inc, "open", now, r.get("value"))
        except RuntimeError as e:
            log(f"Ticket creation FAILED, will retry in {RETRY_SECONDS:.0f} s: {e}")
            self._record(inc, "ticket_failed", now, r.get("value"))
        self._save_state()

    def _resolve(self, key: str, inc: Incident, r: dict, now: float) -> None:
        duration = now - inc.started_ts
        if inc.kind == "camera":
            lines = [f"Readings restored at {r['timestamp']} after {duration:.0f} s."]
        else:
            lines = [f"Back within limits at {r['timestamp']}: {r['value']:g} {inc.unit}.",
                     f"Peak during incident: {inc.peak:g} {inc.unit}. "
                     f"Duration: {duration:.0f} s."]
        log(f"RECOVERED {key} after {duration:.0f} s")
        if inc.ticket is None:
            # Jira was unreachable when the alarm fired: never drop the incident.
            self._open_ticket(inc, r, now)
            if inc.ticket is None:
                with UNSENT_PATH.open("a") as f:
                    f.write(json.dumps({**asdict(inc), "recovery": lines}) + "\n")
                log(f"Jira still unreachable. Incident saved to {UNSENT_PATH} "
                    "- create this ticket manually.")
        if inc.ticket:
            try:
                self.ticketer.comment(inc.ticket, lines)
            except RuntimeError as e:
                log(f"Could not add recovery comment to {inc.ticket}: {e}")
        self._record(inc, "resolved", now, r.get("value"))
        del self.active[key]
        self._save_state()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def make_ticketer(dry_run: bool) -> Ticketer:
    env = {k: os.environ.get(k) for k in
           ("JIRA_URL", "JIRA_EMAIL", "JIRA_API_TOKEN", "JIRA_PROJECT_KEY")}
    if dry_run:
        log("Dry run: tickets will be printed, not created.")
        return DryRunTicketer()
    missing = [k for k, v in env.items() if not v]
    if missing:
        log(f"Jira not configured (missing {', '.join(missing)}). Running as dry run.")
        return DryRunTicketer()
    log(f"Jira: creating tickets in project {env['JIRA_PROJECT_KEY']} at {env['JIRA_URL']}")
    return JiraTicketer(env["JIRA_URL"], env["JIRA_EMAIL"], env["JIRA_API_TOKEN"],
                        env["JIRA_PROJECT_KEY"], os.environ.get("JIRA_ISSUE_TYPE", "Task"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Open Jira tickets from gauge readings.")
    ap.add_argument("--high", type=float, default=30.0,
                    help="alert when the value reaches this or above (default 30)")
    ap.add_argument("--low", type=float, help="alert when the value reaches this or below")
    ap.add_argument("--hold", type=float, default=5.0,
                    help="seconds the value must stay out of limit (default 5)")
    ap.add_argument("--clear-margin", type=float, default=2.0,
                    help="how far back inside the limit before the incident ends (default 2)")
    ap.add_argument("--camera-timeout", type=float, default=30.0,
                    help="seconds of 'unavailable' readings before a camera ticket (default 30)")
    ap.add_argument("--state", type=Path, default=Path("alert_state.json"))
    ap.add_argument("--dry-run", action="store_true", help="print tickets instead of creating")
    ap.add_argument("--no-influx", action="store_true",
                    help="don't record tickets in InfluxDB for the dashboard")
    ap.add_argument("--influx-url", default=os.environ.get("INFLUX_URL", "http://localhost:8086"))
    ap.add_argument("--influx-org", default=os.environ.get("INFLUX_ORG", "gauge-lab"))
    ap.add_argument("--influx-bucket", default=os.environ.get("INFLUX_BUCKET", "gauges"))
    ap.add_argument("--influx-token",
                    default=os.environ.get("INFLUX_TOKEN", "gauge-dev-token-change-me"))
    ap.add_argument("--tickets-file", type=Path, default=Path("tickets.json"),
                    help="where created tickets are remembered (7 days)")
    ap.add_argument("--jira-poll", type=float, default=10.0,
                    help="seconds between Jira status checks for the dashboard (default 10)")
    ap.add_argument("--sync-only", action="store_true",
                    help="only keep the dashboard's Jira ticket statuses up to date "
                         "(no gauge readings, no new tickets)")
    args = ap.parse_args()

    if args.sync_only:
        ticketer = make_ticketer(args.dry_run)
        if isinstance(ticketer, DryRunTicketer):
            sys.exit("--sync-only needs the Jira settings: run 'source jira.env' first.")
        tracker = TicketTracker(args.tickets_file, IncidentLog(
            args.influx_url, args.influx_org, args.influx_bucket, args.influx_token))
        log(f"Sync-only: checking Jira status of {len(tracker.tickets())} ticket(s) "
            f"every {args.jira_poll:g} s. Ctrl+C to stop.")
        try:
            poll_jira_status(tracker, ticketer, args.jira_poll)
        except KeyboardInterrupt:
            pass
        return

    if args.high is None and args.low is None:
        sys.exit("Set at least one limit, e.g. --high 30")

    ticketer = make_ticketer(args.dry_run)
    tracker = None
    if not args.no_influx:
        tracker = TicketTracker(args.tickets_file, IncidentLog(
            args.influx_url, args.influx_org, args.influx_bucket, args.influx_token))
        threading.Thread(target=poll_jira_status, args=(tracker, ticketer, args.jira_poll),
                         daemon=True).start()
    monitor = AlertMonitor(ticketer, args.high, args.low, args.hold,
                           args.clear_margin, args.camera_timeout, args.state, tracker)
    limits = " and ".join(x for x in (
        f">= {args.high:g}" if args.high is not None else "",
        f"<= {args.low:g}" if args.low is not None else "") if x)
    log(f"Watching for values {limits} held {args.hold:g} s. Ctrl+C to stop.")

    try:
        while True:
            line = sys.stdin.readline()
            if not line:
                if tracker is None:
                    log("Input ended (gauge_reader stopped).")
                    break
                log("Input ended (gauge_reader stopped). Still keeping the dashboard's "
                    "Jira ticket statuses up to date - Ctrl+C to stop.")
                while True:
                    time.sleep(3600)
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                reading = json.loads(line)
                if reading.get("type") == "light":
                    continue                      # indicator lights: dashboard only
                monitor.process(reading)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                log(f"Skipping bad line: {e}")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
