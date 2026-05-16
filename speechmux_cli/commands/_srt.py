"""SRT subtitle formatting utilities."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO


def _to_srt_timestamp(seconds: float) -> str:
    """Convert a duration in seconds to SRT timestamp format HH:MM:SS,mmm.

    Args:
        seconds: Non-negative duration in seconds.

    Returns:
        Timestamp string formatted as ``HH:MM:SS,mmm``.
    """
    total_ms = max(0, int(seconds * 1000))
    millis = total_ms % 1000
    total_s = total_ms // 1000
    hours = total_s // 3600
    minutes = (total_s % 3600) // 60
    secs = total_s % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_srt_block(index: int, start_sec: float, end_sec: float, text: str) -> str:
    """Format a single SRT subtitle block.

    Args:
        index: 1-based subtitle index.
        start_sec: Start time in seconds.
        end_sec: End time in seconds. When 0 or less than start_sec,
            a 1-second duration is assumed.
        text: Subtitle text (will be stripped).

    Returns:
        SRT block string including trailing blank line.
    """
    if end_sec <= start_sec:
        end_sec = start_sec + 1.0
    return (
        f"{index}\n"
        f"{_to_srt_timestamp(start_sec)} --> {_to_srt_timestamp(end_sec)}\n"
        f"{text.strip()}\n\n"
    )


@dataclass
class SrtEntry:
    """A single SRT subtitle entry.

    Attributes:
        start_sec: Start time in seconds.
        end_sec: End time in seconds.
        text: Subtitle text.
    """

    start_sec: float
    end_sec: float
    text: str


@dataclass
class SrtWriter:
    """Accumulates final recognition results and writes an SRT file.

    Attributes:
        entries: Collected subtitle entries (only populated with final results).
    """

    entries: list[SrtEntry] = field(default_factory=list)

    def add(self, start_sec: float, end_sec: float, text: str) -> None:
        """Add a final recognition result as a subtitle entry.

        Args:
            start_sec: Start time in seconds.
            end_sec: End time in seconds.
            text: Transcription text.
        """
        stripped = text.strip()
        if stripped:
            self.entries.append(SrtEntry(start_sec=start_sec, end_sec=end_sec, text=stripped))

    def write(self, destination: str | Path) -> None:
        """Write accumulated entries to an SRT file or stdout.

        Args:
            destination: File path, or ``"-"`` to write to stdout.
        """
        output_file: IO[str]
        if str(destination) == "-":
            output_file = sys.stdout
            self._write_to(output_file)
        else:
            with open(destination, "w", encoding="utf-8") as output_file:
                self._write_to(output_file)

    def _write_to(self, output_file: IO[str]) -> None:
        for index, entry in enumerate(self.entries, start=1):
            output_file.write(
                format_srt_block(index, entry.start_sec, entry.end_sec, entry.text)
            )
