"""speechmux file — transcribe a single audio file."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterator

import click
import grpc

from speechmux_cli.audio.loader import AudioLoader, AudioLoadError
from speechmux_cli.client.grpc_client import SpeechMuxClient, SpeechMuxError
from speechmux_cli.commands._output import (
    format_summary,
    print_error,
    print_no_speech,
    print_result,
)
from speechmux_cli.types import ClientOptions


@click.command("file")
@click.argument("audio_path", type=click.Path(exists=False))
@click.option("--realtime", is_flag=True, default=False, help="Pace audio at real-time speed.")
@click.option("--chunk-ms", default=100, show_default=True, help="Audio chunk size in ms.")
@click.option("--metrics", is_flag=True, default=False, help="Print timing metrics after run.")
@click.option("--json", "json_mode", is_flag=True, default=False, help="Output JSON lines.")
@click.option("--lang", "language", default=None, help="BCP-47 language code (overrides global --lang).")
@click.option("--task", type=click.Choice(["transcribe", "translate"]), default=None, help="Task (overrides global --task).")
@click.option("--profile", type=click.Choice(["realtime", "accurate"]), default=None, help="Decode profile (overrides global --profile).")
@click.option("--vad-silence", "vad_silence", default=None, type=float, help="VAD silence duration in seconds (overrides global).")
@click.option("--vad-threshold", "vad_threshold", default=None, type=float, help="VAD speech threshold (overrides global).")
@click.option("--engine-hint", "engine_hint", default=None,
              help="Engine endpoint id (overrides global --engine-hint).")
@click.option(
    "--vad-mode", "vad_mode",
    type=click.Choice(["", "continue", "auto-end"]),
    default=None,
    help="VAD mode (overrides global --vad-mode).",
)
@click.pass_context
def file_cmd(
    click_context: click.Context,
    audio_path: str,
    realtime: bool,
    chunk_ms: int,
    metrics: bool,
    json_mode: bool,
    language: str | None,
    task: str | None,
    profile: str | None,
    vad_silence: float | None,
    vad_threshold: float | None,
    engine_hint: str | None,
    vad_mode: str | None,
) -> None:
    """Transcribe a single audio file (WAV, FLAC, OGG, MP3)."""
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
    if engine_hint is not None:
        client_options["engine_hint"] = engine_hint
    if vad_mode is not None:
        client_options["vad_mode"] = vad_mode

    loader = AudioLoader(audio_path, chunk_ms=chunk_ms)
    try:
        loader.validate()
    except AudioLoadError as audio_load_error:
        print_error(str(audio_load_error))
        sys.exit(1)

    try:
        client = SpeechMuxClient(
            client_options["server"],
            api_key=client_options["api_key"],
            tls=client_options["tls"],
            tls_ca_file=client_options["tls_ca_file"],
            connect_timeout=client_options["connect_timeout"],
        )
    except ConnectionError as connection_error:
        print_error(str(connection_error))
        sys.exit(1)

    committed_so_far = ""
    result_count = 0
    final_text = ""
    last_language = ""
    start_time = time.perf_counter()

    try:
        chunk_duration_sec = chunk_ms / 1000.0

        def _paced_chunks() -> Iterator[bytes]:
            for chunk in loader.chunks():
                yield chunk
                if realtime:
                    time.sleep(chunk_duration_sec)

        with client:
            for result in client.stream(
                _paced_chunks(),
                language=client_options["language"],
                task=client_options["task"],
                profile=client_options["profile"],
                stream_mode="realtime" if realtime else "batch",
                vad_silence=client_options["vad_silence"],
                vad_threshold=client_options["vad_threshold"],
                session_timeout=client_options["session_timeout"],
                engine_hint=client_options["engine_hint"],
                vad_mode=client_options["vad_mode"],
            ):
                result_count += 1
                last_language = result.language_code or last_language
                if result.is_final:
                    final_text = result.text or result.committed_text
                committed_so_far = print_result(
                    result, json_mode=json_mode, committed_so_far=committed_so_far
                )

    except SpeechMuxError as speech_mux_error:
        print()  # end any partial line
        print_error(speech_mux_error.args[0], code=speech_mux_error.code)
        sys.exit(1)
    except grpc.RpcError as rpc_error:
        print()
        print_error(
            f"gRPC error: {rpc_error.details() if hasattr(rpc_error, 'details') else rpc_error}"
        )
        sys.exit(1)

    elapsed_wall_sec = time.perf_counter() - start_time

    if result_count == 0:
        print_no_speech()

    if metrics:
        summary = format_summary(
            audio_path,
            loader.duration_sec,
            elapsed_wall_sec,
            result_count,
            final_text,
            last_language,
        )
        print(json.dumps(summary, ensure_ascii=False))
