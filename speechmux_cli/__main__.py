"""speechmux CLI entrypoint."""

from __future__ import annotations

import click

from speechmux_cli.commands.batch import batch_cmd
from speechmux_cli.commands.file import file_cmd
from speechmux_cli.commands.mic import mic_cmd


@click.group()
@click.option("--server", default="localhost:50051", show_default=True, help="Core gRPC address.")
@click.option("--api-key", default="", envvar="SPEECHMUX_API_KEY", help="API key.")
@click.option(
    "--lang",
    "language",
    default="",
    help="BCP-47 language code (empty = auto-detect).",
)
@click.option(
    "--task",
    type=click.Choice(["transcribe", "translate"]),
    default="transcribe",
    show_default=True,
)
@click.option(
    "--profile",
    type=click.Choice(["realtime", "accurate"]),
    default="realtime",
    show_default=True,
    help="Decode profile.",
)
@click.option("--vad-silence", default=0.8, show_default=True, help="VAD silence duration (s).")
@click.option(
    "--vad-threshold",
    default=0.5,
    show_default=True,
    help="VAD speech probability threshold.",
)
@click.option(
    "--engine-hint",
    "engine_hint",
    default="",
    envvar="SPEECHMUX_ENGINE_HINT",
    help="Route to a specific engine endpoint id (e.g. faster-whisper).",
)
@click.option(
    "--vad-mode",
    "vad_mode",
    type=click.Choice(["", "continue", "auto-end"]),
    default="",
    show_default=False,
    help="VAD session mode: continue (default) or auto-end (close after first utterance).",
)
@click.option("--connect-timeout", default=10.0, show_default=True, help="Connection timeout (s).")
@click.option(
    "--session-timeout",
    default=300.0,
    show_default=True,
    help="Max session duration (s).",
)
@click.option("--tls", is_flag=True, default=False, help="Enable TLS.")
@click.option("--tls-ca-file", default=None, help="Path to CA certificate for TLS.")
@click.pass_context
def cli(
    click_context: click.Context,
    server: str,
    api_key: str,
    language: str,
    task: str,
    profile: str,
    vad_silence: float,
    vad_threshold: float,
    engine_hint: str,
    vad_mode: str,
    connect_timeout: float,
    session_timeout: float,
    tls: bool,
    tls_ca_file: str | None,
) -> None:
    """SpeechMux CLI — transcribe audio via SpeechMux Core."""
    click_context.ensure_object(dict)
    click_context.obj.update(
        {
            "server": server,
            "api_key": api_key,
            "language": language,
            "task": task,
            "profile": profile,
            "vad_silence": vad_silence,
            "vad_threshold": vad_threshold,
            "engine_hint": engine_hint,
            "vad_mode": vad_mode,
            "connect_timeout": connect_timeout,
            "session_timeout": session_timeout,
            "tls": tls,
            "tls_ca_file": tls_ca_file,
        }
    )


cli.add_command(file_cmd)
cli.add_command(batch_cmd)
cli.add_command(mic_cmd)

if __name__ == "__main__":
    cli()
