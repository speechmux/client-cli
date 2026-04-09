"""gRPC client for SpeechMux Core."""

from speechmux_cli.client.grpc_client import SpeechMuxClient, SpeechMuxError, StreamResult

__all__ = ["SpeechMuxClient", "SpeechMuxError", "StreamResult"]
