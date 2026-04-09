"""gRPC client for SpeechMux Core StreamingRecognize."""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import grpc
from stt_proto.client.v1 import client_pb2, client_pb2_grpc

_ERR_CODE_RE = re.compile(r"(ERR\d{4})")

# gRPC status codes that are safe to retry (no audio consumed yet)
_RETRYABLE_GRPC_CODES = {
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
}

# SpeechMux error codes that are safe to retry
_RETRYABLE_ERR_CODES = {"ERR1011", "ERR2008"}


@dataclass(frozen=True)
class StreamResult:
    """A single recognition result from the server.

    Attributes:
        is_final: Whether this result is final (utterance complete).
        text: Final transcript text for final results.
        committed_text: Accumulated committed text so far.
        unstable_text: In-progress unstable hypothesis.
        audio_duration: Duration of audio processed so far, in seconds.
        language_code: Detected or configured BCP-47 language code.
        latency_sec: Server-side processing latency in seconds.
        rtf: Real-time factor (wall time / audio duration).
        utterance_index: Zero-based index of this utterance in the session.
    """

    is_final: bool
    text: str
    committed_text: str
    unstable_text: str
    audio_duration: float
    language_code: str
    latency_sec: float = field(default=0.0)
    rtf: float = field(default=0.0)
    utterance_index: int = field(default=0)

    @property
    def display_text(self) -> str:
        """Best text for display: committed + unstable, or text for finals."""
        if self.committed_text or self.unstable_text:
            return f"{self.committed_text} {self.unstable_text}".strip()
        return self.text


class SpeechMuxError(Exception):
    """Terminal error from the SpeechMux server.

    Attributes:
        code: SpeechMux error code string (e.g. "ERR1001"), empty if unavailable.
        retryable: Whether the client may safely retry the request.
    """

    def __init__(self, message: str, code: str = "", retryable: bool = False) -> None:
        """Initialize SpeechMuxError.

        Args:
            message: Human-readable error description.
            code: SpeechMux error code string (e.g. "ERR1001").
            retryable: Whether the client may safely retry the request.
        """
        super().__init__(message)
        self.code = code
        self.retryable = retryable


def _extract_err_code(rpc_error: grpc.RpcError) -> str:
    try:
        message = getattr(rpc_error, "details", lambda: "")() or ""
    except Exception:
        message = ""
    regex_match = _ERR_CODE_RE.search(message)
    return regex_match.group(1) if regex_match else ""


def _is_retryable_grpc(rpc_error: grpc.RpcError) -> bool:
    grpc_code = rpc_error.code() if hasattr(rpc_error, "code") else None
    if grpc_code in _RETRYABLE_GRPC_CODES:
        return True
    return _extract_err_code(rpc_error) in _RETRYABLE_ERR_CODES


class SpeechMuxClient:
    """Thin gRPC wrapper for SpeechMux Core StreamingRecognize.

    Usage::

        with SpeechMuxClient("localhost:50051") as client:
            for result in client.stream(audio_chunks(), language="ko"):
                print(result.display_text)
    """

    def __init__(
        self,
        server: str = "localhost:50051",
        *,
        api_key: str = "",
        tls: bool = False,
        tls_ca_file: str | None = None,
        connect_timeout: float = 10.0,
        keepalive_time_ms: int = 30_000,
        keepalive_timeout_ms: int = 10_000,
    ) -> None:
        """Initialize and connect the gRPC channel.

        Args:
            server: gRPC server address in host:port format.
            api_key: API key sent in each SessionConfig.
            tls: Enable TLS channel credentials.
            tls_ca_file: Path to PEM CA certificate for TLS verification.
            connect_timeout: Seconds to wait for the channel to become ready.
            keepalive_time_ms: Interval between keepalive pings in milliseconds.
            keepalive_timeout_ms: Keepalive timeout in milliseconds.

        Raises:
            ConnectionError: If the channel is not ready within connect_timeout.
        """
        channel_options = [
            ("grpc.keepalive_time_ms", keepalive_time_ms),
            ("grpc.keepalive_timeout_ms", keepalive_timeout_ms),
            ("grpc.keepalive_permit_without_calls", 1),
            ("grpc.http2.max_pings_without_data", 0),
        ]

        if tls or tls_ca_file:
            root_certs: bytes | None = None
            if tls_ca_file:
                root_certs = Path(tls_ca_file).read_bytes()
            credentials = grpc.ssl_channel_credentials(root_certificates=root_certs)
            self._channel: grpc.Channel = grpc.secure_channel(
                server, credentials, options=channel_options
            )
        else:
            self._channel = grpc.insecure_channel(server, options=channel_options)

        try:
            grpc.channel_ready_future(self._channel).result(timeout=connect_timeout)
        except grpc.FutureTimeoutError:
            self._channel.close()
            raise ConnectionError(
                f"Could not connect to SpeechMux Core at {server} "
                f"within {connect_timeout:.0f}s. Is the server running?"
            ) from None

        self._stub = client_pb2_grpc.STTServiceStub(self._channel)
        self._api_key = api_key

    def close(self) -> None:
        """Close the underlying gRPC channel."""
        self._channel.close()

    def __enter__(self) -> SpeechMuxClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def stream(
        self,
        audio_chunks: Iterable[bytes],
        *,
        session_id: str | None = None,
        language: str = "",
        task: str = "transcribe",
        profile: str = "realtime",
        stream_mode: str = "batch",
        vad_silence: float = 0.8,
        vad_threshold: float = 0.5,
        session_timeout: float = 300.0,
    ) -> Iterator[StreamResult]:
        """Stream audio chunks and yield recognition results.

        Args:
            audio_chunks: Iterable of PCM S16LE byte chunks at 16kHz mono.
            session_id: Optional session identifier; generated if not provided.
            language: BCP-47 language code, empty for auto-detect.
            task: Recognition task, "transcribe" or "translate".
            profile: Decode profile, "realtime" or "accurate".
            stream_mode: Stream mode, "realtime" or "batch".
            vad_silence: VAD silence duration threshold in seconds.
            vad_threshold: VAD speech probability threshold.
            session_timeout: gRPC deadline for the entire session in seconds.

        Yields:
            StreamResult for each result received from the server.

        Raises:
            SpeechMuxError: On terminal server-side errors.
            grpc.RpcError: On transport-level errors.
        """
        resolved_session_id = session_id or str(uuid.uuid4())

        task_enum = (
            client_pb2.TASK_TRANSLATE if task.lower() == "translate" else client_pb2.TASK_TRANSCRIBE
        )
        profile_enum = (
            client_pb2.DECODE_PROFILE_ACCURATE
            if profile.lower() == "accurate"
            else client_pb2.DECODE_PROFILE_REALTIME
        )
        mode_enum = (
            client_pb2.STREAM_MODE_REALTIME
            if stream_mode.lower() == "realtime"
            else client_pb2.STREAM_MODE_BATCH
        )

        def _requests() -> Iterator[client_pb2.StreamingRecognizeRequest]:
            yield client_pb2.StreamingRecognizeRequest(
                session_config=client_pb2.SessionConfig(
                    session_id=resolved_session_id,
                    api_key=self._api_key,
                    stream_mode=mode_enum,
                    audio_config=client_pb2.AudioConfig(
                        encoding=client_pb2.AUDIO_ENCODING_PCM_S16LE,
                        sample_rate=16000,
                        channels=1,
                    ),
                    recognition_config=client_pb2.RecognitionConfig(
                        language_code=language,
                        task=task_enum,
                        decode_profile=profile_enum,
                    ),
                    vad_config=client_pb2.VADConfig(
                        silence_duration=vad_silence,
                        threshold=vad_threshold,
                    ),
                )
            )
            for chunk in audio_chunks:
                yield client_pb2.StreamingRecognizeRequest(audio=chunk)
            yield client_pb2.StreamingRecognizeRequest(signal=client_pb2.StreamSignal(is_last=True))

        responses = self._stub.StreamingRecognize(_requests(), timeout=session_timeout)

        # First response must be session_created or error.
        try:
            first_response = next(iter(responses))
        except StopIteration:
            raise SpeechMuxError("Server closed stream before sending session_created") from None
        except grpc.RpcError as exc:
            retryable = exc.code() in _RETRYABLE_GRPC_CODES
            raise SpeechMuxError(
                str(exc.details()),
                code=_extract_err_code(exc),
                retryable=retryable,
            ) from exc

        if first_response.HasField("error"):
            server_error = first_response.error
            raise SpeechMuxError(
                server_error.message,
                code=server_error.error_code,
                retryable=server_error.retryable,
            )
        if not first_response.HasField("session_created"):
            raise SpeechMuxError(
                f"Unexpected first response type: {first_response.WhichOneof('streaming_response')}"
            )

        for response in responses:
            if response.HasField("error"):
                server_error = response.error
                raise SpeechMuxError(
                    server_error.message,
                    code=server_error.error_code,
                    retryable=server_error.retryable,
                )
            if response.HasField("result"):
                recognition_result = response.result
                result_metadata = recognition_result.meta
                yield StreamResult(
                    is_final=recognition_result.is_final,
                    text=recognition_result.text,
                    committed_text=recognition_result.committed_text,
                    unstable_text=recognition_result.unstable_text,
                    audio_duration=recognition_result.audio_duration,
                    language_code=recognition_result.language_code,
                    latency_sec=result_metadata.latency_sec if result_metadata else 0.0,
                    rtf=result_metadata.real_time_factor if result_metadata else 0.0,
                    utterance_index=result_metadata.utterance_index if result_metadata else 0,
                )
