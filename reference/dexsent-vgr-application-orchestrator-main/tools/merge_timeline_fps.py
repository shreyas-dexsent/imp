#!/usr/bin/env python3
"""Implementation for `tools.merge_timeline_fps`."""

import argparse
import datetime
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

LINE_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} [\d:,]+) - vision_cache - INFO - "
    r"\[vision_cache\] fps request_id=(?P<request_id>\S+) frame_id=(?P<frame_id>\S+) fps=(?P<fps>[\d.]+)"
)


def parse_timeline(path: Path) -> List[Tuple[Optional[datetime.datetime], dict, int]]:
    entries = []
    for idx, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        event = json.loads(line)
        ts: Optional[datetime.datetime] = None
        timestamp_ns = event.get("timestamp_ns") or event.get("vision_timestamp_ns")
        if isinstance(timestamp_ns, (int, float)):
            ts = datetime.datetime.fromtimestamp(timestamp_ns / 1e9)
        entries.append((ts, event, idx))
    return entries


def parse_fps_log(path: Path) -> List[Tuple[Optional[datetime.datetime], dict, int]]:
    entries = []
    for idx, line in enumerate(path.read_text().splitlines()):
        match = LINE_RE.search(line)
        if not match:
            continue
        ts_raw = match.group("ts")
        try:
            ts = datetime.datetime.strptime(ts_raw, "%Y-%m-%d %H:%M:%S,%f")
        except ValueError:
            ts = None
        entries.append(
            (
                ts,
                {
                    "source": "vision_fps",
                    "request_id": match.group("request_id"),
                    "frame_id": match.group("frame_id"),
                    "fps": float(match.group("fps")),
                },
                idx,
            )
        )
    return entries


def merge_entries(
    timeline: Iterable[Tuple[Optional[datetime.datetime], dict, int]],
    fps_logs: Iterable[Tuple[Optional[datetime.datetime], dict, int]],
) -> List[Tuple[Optional[datetime.datetime], str]]:
    combined = []
    for ts, event, order in timeline:
        combined.append(
            (
                ts,
                f"TIMELINE [{order}] {json.dumps(event, separators=(',', ':'))}",
                order * 2,
            )
        )
    fps_offset = len(combined)
    for ts, fps_record, order in fps_logs:
        combined.append(
            (
                ts,
                f"FPS [{order}] {json.dumps(fps_record, separators=(',', ':'))}",
                fps_offset + order,
            )
        )
    combined.sort(key=lambda item: (item[0] or datetime.datetime.min, item[2]))
    return combined


def format_entry(ts: Optional[datetime.datetime], text: str) -> str:
    ts_str = ts.isoformat() if ts else "N/A"
    return f"{ts_str}  {text}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge timeline events with recorded Vision FPS logs."
    )
    parser.add_argument(
        "timeline",
        type=Path,
        help="Path to timeline.jsonl produced by the orchestrator run.",
    )
    parser.add_argument(
        "--fps-log",
        type=Path,
        help="Path to the orchestrator log that contains the vision_cache fps messages.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only print the first LIMIT merged entries (0 = no limit).",
    )

    args = parser.parse_args()
    if not args.timeline.exists():
        raise SystemExit(f"Timeline file not found: {args.timeline}")

    timeline_entries = parse_timeline(args.timeline)
    fps_entries = (
        parse_fps_log(args.fps_log) if args.fps_log and args.fps_log.exists() else []
    )
    merged = merge_entries(timeline_entries, fps_entries)
    if args.limit:
        merged = merged[: args.limit]

    for ts, text in merged:
        print(format_entry(ts, text))


if __name__ == "__main__":
    main()
