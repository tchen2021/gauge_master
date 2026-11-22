#!/usr/bin/env python3
"""
make_dashboard.py - builds the Grafana dashboard (gauge_monitor.json) for any
number of gauges and indicator lights.

  python make_dashboard.py                                  # 2 gauges, 4 lights
  python make_dashboard.py --gauges gauge-1,gauge-2 --lights light-1,light-2,light-3,light-4
  python make_dashboard.py --light-names "light-1=Pump fault,light-2=Overheat"
  python make_dashboard.py --warn 50 --alarm 60 --min 0 --max 100

Write it straight into Grafana's folder (Grafana reloads it within ~10 s):
  python make_dashboard.py --out dashboard/grafana/dashboards/gauge_monitor.json

The gauge/light ids must match the ids in the gauge_reader calibration file.
Keep --alarm equal to alert_monitor.py's --high so the red banner and the
Jira tickets agree.
"""
from __future__ import annotations

import argparse
import json

DS = {"type": "influxdb", "uid": "influxdb"}


def q(measure: str, field: str, tag: str, tid: str, window: str = "-10s", trend: bool = False) -> list:
    rng = "start: v.timeRangeStart, stop: v.timeRangeStop" if trend else f"start: {window}"
    tail = ("  |> aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)"
            if trend else "  |> last()")
    return [{"refId": "A", "datasource": DS, "query":
             f'from(bucket: "gauges")\n  |> range({rng})\n'
             f'  |> filter(fn: (r) => r._measurement == "{measure}" and r._field == "{field}" '
             f'and r.{tag} == "{tid}")\n{tail}'}]


def thresholds(warn: float, alarm: float) -> dict:
    return {"mode": "absolute", "steps": [{"color": "green", "value": None},
                                          {"color": "yellow", "value": warn},
                                          {"color": "red", "value": alarm}]}


def gauge_panels(gid: str, x: int, y: int, w: int, a: argparse.Namespace, pid: int) -> tuple[list, int]:
    half, big = w // 2, w * 5 // 12
    p = []
    p.append({"id": pid, "type": "stat", "title": f"{gid} - Pressure status", "datasource": DS,
              "gridPos": {"h": 4, "w": w, "x": x, "y": y},
              "description": f"WARNING at {a.warn:g}+, HIGH PRESSURE ALARM at {a.alarm:g}+.",
              "fieldConfig": {"defaults": {
                  "unit": "pressurepsi", "decimals": 1, "noValue": "NO DATA - pressure unknown",
                  "displayName": gid, "color": {"mode": "thresholds"},
                  "thresholds": {"mode": "absolute", "steps": [{"color": "#808080", "value": None}]},
                  "mappings": [
                      {"type": "range", "options": {"from": a.alarm, "to": 1e9, "result":
                          {"text": "HIGH PRESSURE ALARM", "color": "red", "index": 0}}},
                      {"type": "range", "options": {"from": a.warn, "to": a.alarm, "result":
                          {"text": "WARNING", "color": "yellow", "index": 1}}},
                      {"type": "range", "options": {"from": -1e9, "to": a.warn, "result":
                          {"text": "NORMAL", "color": "green", "index": 2}}}]},
                  "overrides": []},
              "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                          "colorMode": "background", "graphMode": "none", "textMode": "value_and_name",
                          "justifyMode": "center", "orientation": "auto"},
              "targets": q("gauge", "value", "gauge_id", gid)})
    p.append({"id": pid + 1, "type": "gauge", "title": f"{gid} - Pressure (live)", "datasource": DS,
              "gridPos": {"h": 8, "w": big, "x": x, "y": y + 4},
              "fieldConfig": {"defaults": {"unit": "pressurepsi", "min": a.min, "max": a.max,
                                           "decimals": 1, "thresholds": thresholds(a.warn, a.alarm),
                                           "color": {"mode": "thresholds"}, "displayName": gid,
                                           "noValue": "NO DATA"}, "overrides": []},
              "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                          "showThresholdLabels": True, "showThresholdMarkers": True},
              "targets": q("gauge", "value", "gauge_id", gid)})
    p.append({"id": pid + 2, "type": "timeseries", "title": f"{gid} - Pressure trend", "datasource": DS,
              "gridPos": {"h": 8, "w": w - big, "x": x + big, "y": y + 4},
              "fieldConfig": {"defaults": {"unit": "pressurepsi", "min": a.min, "max": a.max,
                                           "decimals": 1, "thresholds": thresholds(a.warn, a.alarm),
                                           "color": {"mode": "fixed", "fixedColor": "blue"},
                                           "displayName": gid,
                                           "custom": {"lineWidth": 2, "fillOpacity": 10,
                                                      "showPoints": "never", "spanNulls": False,
                                                      "thresholdsStyle": {"mode": "line+area"}}},
                              "overrides": []},
              "options": {"legend": {"displayMode": "list", "placement": "bottom", "showLegend": False},
                          "tooltip": {"mode": "single", "sort": "none"}},
              "targets": q("gauge", "value", "gauge_id", gid, trend=True)})
    p.append({"id": pid + 3, "type": "stat", "title": f"{gid} - Camera / reader status", "datasource": DS,
              "gridPos": {"h": 4, "w": half, "x": x, "y": y + 12},
              "fieldConfig": {"defaults": {
                  "noValue": "NO DATA - reader stopped", "color": {"mode": "thresholds"},
                  "thresholds": {"mode": "absolute", "steps": [{"color": "red", "value": None},
                                                               {"color": "green", "value": 1}]},
                  "mappings": [{"type": "value", "options": {
                      "1": {"text": "READING OK", "color": "green", "index": 0},
                      "0": {"text": "CAN'T READ GAUGE", "color": "red", "index": 1}}}]},
                  "overrides": []},
              "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                          "colorMode": "background", "graphMode": "none", "textMode": "value",
                          "justifyMode": "center"},
              "targets": q("gauge", "ok", "gauge_id", gid)})
    p.append({"id": pid + 4, "type": "stat", "title": f"{gid} - Reader confidence", "datasource": DS,
              "gridPos": {"h": 4, "w": w - half, "x": x + half, "y": y + 12},
              "fieldConfig": {"defaults": {
                  "unit": "percentunit", "min": 0, "max": 1, "decimals": 0, "noValue": "NO DATA",
                  "color": {"mode": "thresholds"},
                  "thresholds": {"mode": "absolute", "steps": [{"color": "red", "value": None},
                                                               {"color": "yellow", "value": 0.3},
                                                               {"color": "green", "value": 0.5}]}},
                  "overrides": []},
              "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                          "colorMode": "value", "graphMode": "area", "textMode": "value",
                          "justifyMode": "center"},
              "targets": q("gauge", "confidence", "gauge_id", gid)})
    return p, 16


def light_panel(lid: str, name: str, x: int, y: int, w: int, pid: int) -> dict:
    title = f"{lid} - {name}" if name else lid
    return {"id": pid, "type": "stat", "title": title, "datasource": DS,
            "description": "Indicator light on the board, read by the camera. "
                           "NO DATA = the camera can't see it.",
            "gridPos": {"h": 4, "w": w, "x": x, "y": y},
            "fieldConfig": {"defaults": {
                "noValue": "NO DATA", "color": {"mode": "thresholds"},
                "thresholds": {"mode": "absolute", "steps": [{"color": "#808080", "value": None}]},
                "mappings": [{"type": "value", "options": {
                    "1": {"text": "ON", "color": "red", "index": 0},
                    "0": {"text": "OFF", "color": "#4a4a4a", "index": 1}}}]},
                "overrides": []},
            "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                        "colorMode": "background", "graphMode": "none", "textMode": "value",
                        "justifyMode": "center"},
            "targets": q("light", "on", "light_id", lid)}


def tickets_panel(y: int, pid: int) -> dict:
    def ov(name, *props):
        return {"matcher": {"id": "byName", "options": name},
                "properties": [{"id": k, "value": v} for k, v in props]}
    query = '''from(bucket: "gauges")
  |> range(start: -7d)
  |> filter(fn: (r) => r._measurement == "incident")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> group(columns: ["ticket"])
  |> sort(columns: ["_time"])
  |> last(column: "_time")
  |> group()
  |> sort(columns: ["_time"], desc: true)
  |> limit(n: 20)
  |> keep(columns: ["_time", "gauge_id", "kind", "ticket", "state", "pressure", "peak", "duration_s", "started", "jira_status", "jira_category", "url"])'''
    return {
        "id": pid, "type": "table", "title": "Jira tickets (created by alert_monitor.py)",
        "description": "One row per Jira ticket from the last 7 days, newest first. Click a ticket number "
                       "to open it in Jira. Ticket status: OPEN until moved to a Done-type status in Jira "
                       "(checked every 10 s), then CLOSED. Pressure: Pressure High while above the limit, "
                       "Pressure Low once back below it.",
        "datasource": DS, "gridPos": {"h": 8, "w": 24, "x": 0, "y": y},
        "targets": [{"refId": "A", "datasource": DS, "query": query}],
        "options": {"showHeader": True, "cellHeight": "md", "footer": {"show": False},
                    "sortBy": [{"displayName": "Last update", "desc": True}]},
        "transformations": [{"id": "organize", "options": {"indexByName": {
            "_time": 0, "ticket": 1, "jira_category": 2, "pressure": 3, "gauge_id": 4, "kind": 5,
            "peak": 6, "duration_s": 7, "started": 8, "state": 9, "jira_status": 10, "url": 11}}}],
        "fieldConfig": {"defaults": {"custom": {"align": "auto", "cellOptions": {"type": "auto"}},
                                     "noValue": "No tickets in the last 7 days"},
                        "overrides": [
            ov("url", ("custom.hidden", True)),
            ov("state", ("custom.hidden", True)),
            ov("jira_status", ("custom.hidden", True)),
            ov("_time", ("displayName", "Last update")),
            ov("gauge_id", ("displayName", "Gauge")),
            ov("started", ("displayName", "Started (UTC)")),
            ov("ticket", ("displayName", "Jira ticket"),
               ("links", [{"title": "Open in Jira", "url": "${jira_site}/browse/${__value.raw}",
                           "targetBlank": True}])),
            ov("kind", ("displayName", "Type"),
               ("mappings", [{"type": "value", "options": {
                   "high": {"text": "High pressure", "index": 0},
                   "low": {"text": "Low pressure", "index": 1},
                   "camera": {"text": "Camera can't read gauge", "index": 2}}}])),
            ov("jira_category", ("displayName", "Ticket status"), ("noValue", "checking..."),
               ("custom.cellOptions", {"type": "color-background", "mode": "basic"}),
               ("mappings", [{"type": "value", "options": {
                   "new": {"text": "OPEN", "color": "red", "index": 0},
                   "indeterminate": {"text": "OPEN", "color": "red", "index": 1},
                   "done": {"text": "CLOSED", "color": "green", "index": 2},
                   "deleted": {"text": "DELETED", "color": "#808080", "index": 3}}}])),
            ov("pressure", ("displayName", "Pressure"), ("noValue", "-"),
               ("custom.cellOptions", {"type": "color-background", "mode": "basic"}),
               ("mappings", [{"type": "value", "options": {
                   "Pressure High": {"color": "red", "index": 0},
                   "Pressure Low": {"color": "green", "index": 1},
                   "Camera Fault": {"color": "red", "index": 2},
                   "Camera OK": {"color": "green", "index": 3}}}])),
            ov("peak", ("displayName", "Peak"), ("unit", "pressurepsi"), ("decimals", 1)),
            ov("duration_s", ("displayName", "Duration"), ("unit", "s"), ("decimals", 0)),
        ]}}


def row(title: str, y: int, pid: int) -> dict:
    return {"id": pid, "type": "row", "title": title, "collapsed": False, "panels": [],
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": y}}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the Grafana gauge-monitor dashboard.")
    ap.add_argument("--gauges", default="gauge-1,gauge-2")
    ap.add_argument("--lights", default="light-1,light-2,light-3,light-4")
    ap.add_argument("--light-names", default="", help='e.g. "light-1=Pump fault,light-2=Overheat"')
    ap.add_argument("--warn", type=float, default=50)
    ap.add_argument("--alarm", type=float, default=60)
    ap.add_argument("--min", type=float, default=0)
    ap.add_argument("--max", type=float, default=100)
    ap.add_argument("--jira-site", default="https://sjgggggg.atlassian.net")
    ap.add_argument("--out", default="gauge_monitor.json")
    a = ap.parse_args()

    gauges = [g.strip() for g in a.gauges.split(",") if g.strip()]
    lights = [l.strip() for l in a.lights.split(",") if l.strip()]
    names = dict(kv.split("=", 1) for kv in a.light_names.split(",") if "=" in kv)

    panels, pid, y = [], 1, 0
    cols = 2 if len(gauges) > 1 else 1
    w = 24 // cols
    for i in range(0, len(gauges), cols):                     # gauges side by side, 2 per row
        panels.append(row("  |  ".join(gauges[i:i + cols]), y, pid)); pid += 1; y += 1
        height = 0
        for j, gid in enumerate(gauges[i:i + cols]):
            ps, height = gauge_panels(gid, j * w, y, w, a, pid)
            panels += ps; pid += len(ps)
        y += height
    if lights:
        panels.append(row("Indicator lights", y, pid)); pid += 1; y += 1
        lw = max(3, 24 // len(lights))
        for k, lid in enumerate(lights):
            panels.append(light_panel(lid, names.get(lid, ""), (k * lw) % 24, y + (k * lw) // 24 * 4,
                                      lw, pid)); pid += 1
        y += 4 * ((len(lights) * lw + 23) // 24)
    panels.append(row("Tickets and alerts", y, pid)); pid += 1; y += 1
    panels.append(tickets_panel(y, pid)); pid += 1; y += 8
    panels.append({"id": pid, "type": "alertlist",
                   "title": "Grafana alert rules (display only - tickets come from alert_monitor.py)",
                   "gridPos": {"h": 7, "w": 24, "x": 0, "y": y},
                   "options": {"viewMode": "list", "groupMode": "default", "groupBy": [], "maxItems": 20,
                               "sortOrder": 1, "dashboardAlerts": False, "alertName": "",
                               "alertInstanceLabelFilter": "", "datasource": "grafana",
                               "showInstances": True,
                               "stateFilter": {"firing": True, "pending": True, "noData": True,
                                               "error": True, "normal": False, "recovering": True}}})

    dash = {"uid": "gauge-monitor", "title": "Gauge Monitor", "tags": ["gauge"], "editable": True,
            "schemaVersion": 39, "version": 1, "refresh": "1s",
            "time": {"from": "now-15m", "to": "now"},
            "timepicker": {"refresh_intervals": ["1s", "5s", "10s", "30s", "1m"]},
            "annotations": {"list": []}, "panels": panels,
            "templating": {"list": [{"type": "constant", "name": "jira_site", "label": "Jira site",
                                     "query": a.jira_site, "hide": 2, "skipUrlSync": True,
                                     "current": {"text": a.jira_site, "value": a.jira_site}}]}}
    with open(a.out, "w") as f:
        json.dump(dash, f, indent=2)
    print(f"Wrote {a.out}: {len(gauges)} gauge(s), {len(lights)} light(s), "
          f"warning {a.warn:g}, alarm {a.alarm:g}")


if __name__ == "__main__":
    main()
