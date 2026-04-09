"""speechmux mic — real-time microphone transcription."""

from __future__ import annotations

import queue
import signal
import sys
import threading
from collections.abc import Generator

import click
import grpc

from speechmux_cli.client.grpc_client import SpeechMuxClient, SpeechMuxError
from speechmux_cli.commands._output import print_error, print_no_speech, print_result
from speechmux_cli.types import ClientOptions

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
CHUNK_MS = 80  # mic chunk size — smaller for lower latency


@click.command("mic")
@click.option("--device", default=None, help="Input device name or index.")
@click.option("--json", "json_mode", is_flag=True, default=False, help="Output JSON lines.")
@click.option("--lang", "language", default=None, help="BCP-47 language code (overrides global --lang).")
@click.option("--task", type=click.Choice(["transcribe", "translate"]), default=None, help="Task (overrides global --task).")
@click.option("--profile", type=click.Choice(["realtime", "accurate"]), default=None, help="Decode profile (overrides global --profile).")
@click.option("--vad-silence", "vad_silence", default=None, type=float, help="VAD silence duration in seconds (overrides global).")
@click.option("--vad-threshold", "vad_threshold", default=None, type=float, help="VAD speech threshold (overrides global).")
@click.pass_context
def mic_cmd(
    click_context: click.Context,
    device: str | None,
    json_mode: bool,
    language: str | None,
    task: str | None,
    profile: str | None,
    vad_silence: float | None,
    vad_threshold: float | None,
) -> None:
    """Transcribe microphone input in real time. Press Ctrl+C to stop."""
    try:
        import sounddevice as sd  # type: ignore[import-untyped]
    except ImportError:
        print_error(
            "sounddevice is required for mic mode. Install with: pip install speechmux-cli[mic]"
        )
        sys.exit(1)

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

    audio_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=200)
    stop_event = threading.Event()

    def _sd_callback(
        indata: object,
        frames: int,  # noqa: ARG001
        time_info: object,  # noqa: ARG001
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            click.echo(f"\r[warn] sounddevice: {status}\033[K", err=True, nl=False)
        if stop_event.is_set():
            raise sd.CallbackStop()
        import numpy as np

        audio_queue.put(
            bytes(np.frombuffer(indata, dtype=np.int16))  # type: ignore[call-overload]
        )

    def _audio_gen() -> Generator[bytes, None, None]:
        """Yield PCM bytes from mic until stop_event is set and queue is drained."""
        while True:
            try:
                chunk = audio_queue.get(timeout=0.5)
            except queue.Empty:
                if stop_event.is_set():
                    return
                continue
            if chunk is None:
                return
            yield chunk
            if stop_event.is_set() and audio_queue.empty():
                return

    # Handle Ctrl+C gracefully: signal stop and close the gRPC channel so that
    # the background stream thread exits immediately rather than waiting for the
    # server to finish processing the remaining audio.
    original_sigint = signal.getsignal(signal.SIGINT)

    def _sigint_handler(sig: int, frame: object) -> None:  # noqa: ARG001
        click.echo("\r[stopping…]", err=True)
        stop_event.set()
        audio_queue.put(None)  # unblock _audio_gen
        # Close the channel to cancel the in-flight StreamingRecognize RPC.
        # The stream thread catches grpc.RpcError but ignores it when
        # stop_event is set, so this produces a clean exit.
        client.close()
        # Restore original handler so a second Ctrl+C force-quits.
        signal.signal(signal.SIGINT, original_sigint)

    signal.signal(signal.SIGINT, _sigint_handler)

    click.echo("Listening… (Ctrl+C to stop)", err=True)

    committed_so_far = ""
    result_count = 0
    stream_error: list[str] = []

    def _stream_thread() -> None:
        nonlocal committed_so_far, result_count
        try:
            with client:
                for result in client.stream(
                    _audio_gen(),
                    language=client_options["language"],
                    task=client_options["task"],
                    profile=client_options["profile"],
                    stream_mode="realtime",
                    vad_silence=client_options["vad_silence"],
                    vad_threshold=client_options["vad_threshold"],
                    session_timeout=client_options["session_timeout"],
                ):
                    result_count += 1
                    committed_so_far = print_result(
                        result, json_mode=json_mode, committed_so_far=committed_so_far
                    )
        except SpeechMuxError as speech_mux_error:
            stream_error.append(f"{speech_mux_error.args[0]} [{speech_mux_error.code}]")
        except grpc.RpcError as rpc_error:
            if not stop_event.is_set():
                stream_error.append(
                    f"gRPC: {rpc_error.details() if hasattr(rpc_error, 'details') else rpc_error}"
                )

    # Open mic stream and run gRPC streaming in a background thread.
    try:
        chunk_frames = SAMPLE_RATE * CHUNK_MS // 1000
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=chunk_frames,
            callback=_sd_callback,
            device=device,
        ):
            grpc_stream_thread = threading.Thread(target=_stream_thread, daemon=True)
            grpc_stream_thread.start()
            grpc_stream_thread.join()
    except Exception as caught_exception:  # noqa: BLE001
        # sounddevice errors: device not found, permission denied, etc.
        stop_event.set()
        print_error(f"Microphone error: {caught_exception}")
        if "No Default Input" in str(caught_exception) or "Invalid device" in str(caught_exception):
            print_error("Check that a microphone is connected and permissions are granted.")
        sys.exit(1)
    finally:
        stop_event.set()
        signal.signal(signal.SIGINT, original_sigint)

    if stream_error:
        print()
        print_error(stream_error[0])
        sys.exit(1)

    if result_count == 0:
        print_no_speech()
