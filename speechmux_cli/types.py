"""Shared type definitions for speechmux_cli."""

from __future__ import annotations

from typing import TypedDict


class ClientOptions(TypedDict):
    """Options populated by the CLI group and passed to subcommands via Click context.

    Attributes:
        server: gRPC server address in host:port format.
        api_key: API key for authentication.
        language: BCP-47 language code, or empty string for auto-detect.
        task: Recognition task, either "transcribe" or "translate".
        profile: Decode profile, either "realtime" or "accurate".
        vad_silence: VAD silence duration threshold in seconds.
        vad_threshold: VAD speech probability threshold.
        connect_timeout: Maximum time in seconds to wait for gRPC connection.
        session_timeout: Maximum session duration in seconds.
        tls: Whether to enable TLS.
        tls_ca_file: Path to CA certificate file for TLS, or None.
        engine_hint: Endpoint id for routing hint; empty string means no hint.
        vad_mode: VAD session mode: "continue", "auto-end", or "" (unspecified).
    """

    server: str
    api_key: str
    language: str
    task: str
    profile: str
    vad_silence: float
    vad_threshold: float
    connect_timeout: float
    session_timeout: float
    tls: bool
    tls_ca_file: str | None
    engine_hint: str
    vad_mode: str
