"""speechmux batch — transcribe a directory of audio files in parallel."""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import click
import grpc

from speechmux_cli.audio.loader import AudioLoader, AudioLoadError
from speechmux_cli.client.grpc_client import SpeechMuxClient, SpeechMuxError
from speechmux_cli.commands._output import format_summary, print_error
from speechmux_cli.types import ClientOptions

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".opus"}


@dataclass
class FileResult:
    """Result of processing a single audio file in a batch run.

    Attributes:
        path: Absolute path to the source audio file.
        status: Outcome: "ok", "failed", or "skipped".
        text: Final transcription text (empty if not ok).
        language_code: Detected or configured language code.
        duration_sec: Duration of the audio in seconds.
        wall_sec: Elapsed processing time in seconds.
        result_count: Number of recognition results received.
        error: Error message if status is "failed".
        extra: Full summary dictionary as returned by format_summary.
    """

    path: str
    status: str  # "ok" | "failed" | "skipped"
    text: str = ""
    language_code: str = ""
    duration_sec: float = 0.0
    wall_sec: float = 0.0
    result_count: int = 0
    error: str = ""
    extra: dict[str, str | float | int] = field(default_factory=dict)


def _collect_files(directory: str) -> list[Path]:
    root = Path(directory)
    if not root.exists():
        raise click.BadParameter(f"Directory not found: {directory}", param_hint="DIRECTORY")
    if not root.is_dir():
        raise click.BadParameter(f"Not a directory: {directory}", param_hint="DIRECTORY")
    files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return files


def _process_file(
    path: Path,
    *,
    client_options: ClientOptions,
    output_dir: Path | None,
    chunk_ms: int,
    on_error: str,  # noqa: ARG001
) -> FileResult:
    result_path = output_dir / (path.stem + ".json") if output_dir else None

    # Resume: skip if result already exists.
    if result_path and result_path.exists():
        return FileResult(path=str(path), status="skipped")

    loader = AudioLoader(str(path), chunk_ms=chunk_ms)
    try:
        loader.validate()
    except AudioLoadError as audio_load_error:
        return FileResult(path=str(path), status="failed", error=str(audio_load_error))

    # Dynamic timeout: audio length × 3 + 30s, capped at session_timeout.
    dynamic_timeout = min(
        client_options["session_timeout"],
        loader.duration_sec * 3 + 30,
    )

    try:
        client = SpeechMuxClient(
            client_options["server"],
            api_key=client_options["api_key"],
            tls=client_options["tls"],
            tls_ca_file=client_options["tls_ca_file"],
            connect_timeout=client_options["connect_timeout"],
        )
    except ConnectionError as connection_error:
        return FileResult(path=str(path), status="failed", error=str(connection_error))

    committed_text = ""
    final_text = ""
    last_language = ""
    result_count = 0
    start_time = time.perf_counter()

    try:
        with client:
            for stream_result in client.stream(
                loader.chunks(),
                language=client_options["language"],
                task=client_options["task"],
                profile=client_options["profile"],
                stream_mode="batch",
                vad_silence=client_options["vad_silence"],
                vad_threshold=client_options["vad_threshold"],
                session_timeout=dynamic_timeout,
            ):
                result_count += 1
                last_language = stream_result.language_code or last_language
                if stream_result.committed_text:
                    committed_text = stream_result.committed_text
                if stream_result.is_final:
                    final_text = stream_result.text or stream_result.committed_text

    except SpeechMuxError as speech_mux_error:
        return FileResult(
            path=str(path),
            status="failed",
            error=f"{speech_mux_error.args[0]} [{speech_mux_error.code}]",
            duration_sec=loader.duration_sec,
            wall_sec=time.perf_counter() - start_time,
        )
    except grpc.RpcError as rpc_error:
        return FileResult(
            path=str(path),
            status="failed",
            error=f"gRPC: {rpc_error.details() if hasattr(rpc_error, 'details') else rpc_error}",
            duration_sec=loader.duration_sec,
            wall_sec=time.perf_counter() - start_time,
        )

    elapsed_wall_sec = time.perf_counter() - start_time
    transcription_text = final_text or committed_text

    summary = format_summary(
        str(path),
        loader.duration_sec,
        elapsed_wall_sec,
        result_count,
        transcription_text,
        last_language,
    )

    if result_path:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    return FileResult(
        path=str(path),
        status="ok",
        text=transcription_text,
        language_code=last_language,
        duration_sec=loader.duration_sec,
        wall_sec=elapsed_wall_sec,
        result_count=result_count,
        extra=summary,
    )


@click.command("batch")
@click.argument("directory", type=click.Path(exists=False))
@click.option("--output", "-o", default=None, help="Output directory for per-file JSON results.")
@click.option("--workers", default=4, show_default=True, help="Concurrent sessions.")
@click.option("--chunk-ms", default=100, show_default=True, help="Audio chunk size in ms.")
@click.option(
    "--on-error",
    type=click.Choice(["continue", "stop"]),
    default="continue",
    show_default=True,
    help="What to do when a file fails.",
)
@click.option("--resume", is_flag=True, default=False, help="Skip files with existing output.")
@click.option("--json", "json_mode", is_flag=True, default=False, help="Output JSON lines.")
@click.option("--lang", "language", default=None, help="BCP-47 language code (overrides global --lang).")
@click.option("--task", type=click.Choice(["transcribe", "translate"]), default=None, help="Task (overrides global --task).")
@click.option("--profile", type=click.Choice(["realtime", "accurate"]), default=None, help="Decode profile (overrides global --profile).")
@click.option("--vad-silence", "vad_silence", default=None, type=float, help="VAD silence duration in seconds (overrides global).")
@click.option("--vad-threshold", "vad_threshold", default=None, type=float, help="VAD speech threshold (overrides global).")
@click.pass_context
def batch_cmd(
    click_context: click.Context,
    directory: str,
    output: str | None,
    workers: int,
    chunk_ms: int,
    on_error: str,
    resume: bool,  # noqa: ARG001
    json_mode: bool,
    language: str | None,
    task: str | None,
    profile: str | None,
    vad_silence: float | None,
    vad_threshold: float | None,
) -> None:
    """Transcribe all audio files in DIRECTORY in parallel."""
    client_options: ClientOptions = dict(click_context.obj)
    if language is not None:
        client_options["language"] = language
    if task is not None:
        client_options["task"] = task
    if profile is not None:
        client_options["profile"] = profile
    if vad_silence is not None:
        client_options["vad_silence"] = vad_silence
    if vad_threshold is not None:
        client_options["vad_threshold"] = vad_threshold

    try:
        files = _collect_files(directory)
    except click.BadParameter as bad_parameter_error:
        print_error(str(bad_parameter_error))
        sys.exit(1)

    if not files:
        print_error(f"No supported audio files found in {directory}")
        sys.exit(1)

    output_dir = Path(output) if output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    total = len(files)
    succeeded = skipped = failed = 0
    abort = False

    click.echo(f"Processing {total} files with {workers} workers...", err=True)

    futures_map = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for path in files:
            if abort:
                break
            future = executor.submit(
                _process_file,
                path,
                client_options=client_options,
                output_dir=output_dir,
                chunk_ms=chunk_ms,
                on_error=on_error,
            )
            futures_map[future] = path

        try:
            for future in as_completed(futures_map):
                file_result = future.result()

                if file_result.status == "ok":
                    succeeded += 1
                    if json_mode:
                        print(json.dumps(file_result.extra, ensure_ascii=False))
                    else:
                        real_time_factor = (
                            file_result.wall_sec / file_result.duration_sec
                            if file_result.duration_sec > 0
                            else 0
                        )
                        click.echo(
                            f"[ok] {file_result.path}  "
                            f"rtf={real_time_factor:.2f}  "
                            f"{file_result.text[:60]}{'…' if len(file_result.text) > 60 else ''}"
                        )
                elif file_result.status == "skipped":
                    skipped += 1
                    if not json_mode:
                        click.echo(f"[skip] {file_result.path}", err=True)
                else:
                    failed += 1
                    print_error(f"[fail] {file_result.path}: {file_result.error}")
                    if on_error == "stop":
                        abort = True
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
        except KeyboardInterrupt:
            click.echo("\nInterrupted — cancelling in-flight requests...", err=True)
            executor.shutdown(wait=False, cancel_futures=True)
            sys.exit(130)

    # Summary.
    summary_line = (
        f"\nDone: {succeeded} succeeded, {skipped} skipped, {failed} failed (total {total})"
    )
    click.echo(summary_line, err=True)

    if failed > 0 and on_error == "stop":
        sys.exit(1)
    if failed > 0:
        sys.exit(2)  # partial failure
