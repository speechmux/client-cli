"""Shared output formatting for CLI commands."""

from __future__ import annotations

import json
import sys

from speechmux_cli.client.grpc_client import StreamResult


def _fmt_ts(seconds: float, minutes_width: int = 1) -> str:
    total_s = int(seconds)
    minutes = total_s // 60
    secs = total_s % 60
    millis = int(round((seconds % 1) * 1000))
    return f"{minutes:0{minutes_width}d}:{secs:02d}.{millis:03d}"


def ts_minutes_width(total_sec: float) -> int:
    """Return the minutes field width needed to display timestamps for a given total duration.

    Args:
        total_sec: Total audio duration in seconds.

    Returns:
        Number of digits required to represent the maximum minute value.
    """
    total_minutes = max(1, int(total_sec / 60))
    return len(str(total_minutes))


def print_result(
    result: StreamResult,
    *,
    json_mode: bool = False,
    committed_so_far: str = "",
    min_width: int = 1,
) -> str:
    """Print a recognition result to stdout and return the updated committed text.

    In terminal mode, partial results overwrite the current line using ANSI
    escape sequences; final results print a newline. In JSON mode, each result
    is emitted as a single JSON line.

    Args:
        result: The recognition result to print.
        json_mode: If True, emit JSON lines instead of terminal output.
        committed_so_far: The accumulated committed text from previous results.

    Returns:
        The updated committed_so_far string after processing this result.
    """
    committed = result.committed_text or committed_so_far
    unstable = result.unstable_text
    kind = "FINAL  " if result.is_final else "partial"

    if json_mode:
        output_object = {
            "is_final": result.is_final,
            "committed_text": result.committed_text,
            "unstable_text": result.unstable_text,
            "text": result.text,
            "language_code": result.language_code,
            "audio_duration": result.audio_duration,
            "latency_sec": result.latency_sec,
            "rtf": result.rtf,
            "utterance_index": result.utterance_index,
            "start_sec": result.start_sec,
            "end_sec": result.end_sec,
        }
        print(json.dumps(output_object, ensure_ascii=False))
    else:
        language_tag = f"[{result.language_code}]" if result.language_code else ""
        if result.is_final:
            final_display = result.text or f"{committed} {unstable}".strip()
            ts_tag = ""
            if result.end_sec > 0:
                ts_start = _fmt_ts(result.start_sec, min_width)
                ts_end = _fmt_ts(result.end_sec, min_width)
                ts_tag = f" [{ts_start} → {ts_end}]"
            print(f"[{kind}]{ts_tag} {language_tag} {final_display}", flush=True)
        else:
            # Show committed in normal weight, unstable dimmed.
            partial_display = committed
            if unstable:
                partial_display = f"{committed} \033[2m{unstable}\033[0m".strip()
            # Overwrite current line.
            print(
                f"\r[{kind}] {language_tag} {partial_display}\033[K",
                end="",
                flush=True,
            )

    if result.is_final and not json_mode:
        print()  # newline after final

    return committed if result.is_final else committed_so_far


def print_no_speech() -> None:
    """Print a message indicating no speech was detected."""
    print("(no speech detected)", file=sys.stderr)


def print_error(message: str, code: str = "") -> None:
    """Print an error message to stderr.

    Args:
        message: The error message to display.
        code: Optional error code to append in brackets.
    """
    suffix = f" [{code}]" if code else ""
    print(f"error: {message}{suffix}", file=sys.stderr)


def format_summary(
    path: str,
    duration_sec: float,
    wall_sec: float,
    result_count: int,
    final_text: str,
    language_code: str | None = None,
) -> dict[str, str | float | int]:
    """Build a summary dictionary for a completed transcription.

    Args:
        path: Path to the audio file.
        duration_sec: Duration of the audio in seconds.
        wall_sec: Elapsed wall-clock time in seconds.
        result_count: Number of recognition results received.
        final_text: The final transcription text.
        language_code: Detected or configured language code.

    Returns:
        Dictionary with file, duration_sec, wall_sec, rtf, results, text,
        and language_code keys.
    """
    real_time_factor = wall_sec / duration_sec if duration_sec > 0 else 0.0
    return {
        "file": path,
        "duration_sec": round(duration_sec, 3),
        "wall_sec": round(wall_sec, 3),
        "rtf": round(real_time_factor, 3),
        "results": result_count,
        "text": final_text,
        "language_code": language_code or "",
    }
