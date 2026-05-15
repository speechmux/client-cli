"""Tests for grpc_client module: StreamResult, helpers, and SpeechMuxClient."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import grpc
import pytest

from speechmux_cli.client.grpc_client import (
    SpeechMuxClient,
    SpeechMuxError,
    StreamResult,
    _extract_err_code,
    _is_retryable_grpc,
    _vad_mode_enum,
)
from stt_proto.client.v1 import client_pb2


# ── StreamResult.display_text ─────────────────────────────────────────────────


def test_display_text_final_with_text():
    """Final result with only text must return text directly."""
    result = StreamResult(
        is_final=True,
        text="hello world",
        committed_text="",
        unstable_text="",
        audio_duration=1.0,
        language_code="en",
    )
    assert result.display_text == "hello world"


def test_display_text_partial_committed_and_unstable():
    """Partial result with committed+unstable must join them with a space."""
    result = StreamResult(
        is_final=False,
        text="",
        committed_text="hello",
        unstable_text="world",
        audio_duration=0.5,
        language_code="en",
    )
    assert result.display_text == "hello world"


def test_display_text_partial_committed_only():
    """Partial result with only committed text must return it stripped."""
    result = StreamResult(
        is_final=False,
        text="ignored",
        committed_text="committed",
        unstable_text="",
        audio_duration=0.5,
        language_code="en",
    )
    assert result.display_text == "committed"


def test_display_text_partial_unstable_only():
    """Partial result with only unstable text must return it stripped."""
    result = StreamResult(
        is_final=False,
        text="ignored",
        committed_text="",
        unstable_text="unstable",
        audio_duration=0.5,
        language_code="en",
    )
    assert result.display_text == "unstable"


def test_display_text_all_empty():
    """Result with no text fields must return an empty string."""
    result = StreamResult(
        is_final=False,
        text="",
        committed_text="",
        unstable_text="",
        audio_duration=0.0,
        language_code="",
    )
    assert result.display_text == ""


def test_display_text_final_fallback_when_committed_and_unstable_empty():
    """Final result with empty committed/unstable must return text field."""
    result = StreamResult(
        is_final=True,
        text="final transcript",
        committed_text="",
        unstable_text="",
        audio_duration=2.0,
        language_code="ko",
    )
    assert result.display_text == "final transcript"


# ── _extract_err_code ─────────────────────────────────────────────────────────


def _make_rpc_error(details_str: str, code: grpc.StatusCode = grpc.StatusCode.INTERNAL) -> MagicMock:
    """Build a mock gRPC RpcError with the given details string.

    Args:
        details_str: String returned by the ``details()`` method.
        code: gRPC status code returned by ``code()``.

    Returns:
        Configured MagicMock that mimics grpc.RpcError.
    """
    # Real gRPC errors implement both grpc.RpcError and grpc.Call; details() and
    # code() live on grpc.Call, so spec=grpc.RpcError alone would omit them.
    err = MagicMock()
    err.details.return_value = details_str
    err.code.return_value = code
    return err


def test_extract_err_code_finds_code():
    """ERR1001 embedded in details must be extracted."""
    err = _make_rpc_error("request rejected: ERR1001 bad api key")
    assert _extract_err_code(err) == "ERR1001"


def test_extract_err_code_no_match_returns_empty():
    """Details string without an ERR code must return empty string."""
    err = _make_rpc_error("something went wrong, no code here")
    assert _extract_err_code(err) == ""


def test_extract_err_code_empty_details():
    """Empty details string must return empty string."""
    err = _make_rpc_error("")
    assert _extract_err_code(err) == ""


def test_extract_err_code_details_is_none():
    """details() returning None must not raise and must return empty string."""
    err = MagicMock()  # no spec — grpc.RpcError lacks details()/code()
    err.details.return_value = None
    assert _extract_err_code(err) == ""


def test_extract_err_code_four_digit_boundary():
    """Only ERR followed by exactly 4 digits must match."""
    err = _make_rpc_error("code ERR9999 end")
    assert _extract_err_code(err) == "ERR9999"


# ── _is_retryable_grpc ────────────────────────────────────────────────────────


def test_is_retryable_unavailable():
    """UNAVAILABLE status code must be retryable."""
    err = _make_rpc_error("service unavailable", grpc.StatusCode.UNAVAILABLE)
    assert _is_retryable_grpc(err) is True


def test_is_retryable_resource_exhausted():
    """RESOURCE_EXHAUSTED status code must be retryable."""
    err = _make_rpc_error("rate limit", grpc.StatusCode.RESOURCE_EXHAUSTED)
    assert _is_retryable_grpc(err) is True


def test_is_retryable_not_found_false():
    """NOT_FOUND status code must not be retryable."""
    err = _make_rpc_error("not found", grpc.StatusCode.NOT_FOUND)
    assert _is_retryable_grpc(err) is False


def test_is_retryable_internal_false():
    """INTERNAL status code must not be retryable by default."""
    err = _make_rpc_error("internal error", grpc.StatusCode.INTERNAL)
    assert _is_retryable_grpc(err) is False


def test_is_retryable_by_err_code():
    """ERR1011 in details must make the error retryable regardless of gRPC code."""
    err = _make_rpc_error("plugin busy ERR1011", grpc.StatusCode.INTERNAL)
    assert _is_retryable_grpc(err) is True


def test_is_retryable_err2008_by_code():
    """ERR2008 in details must make the error retryable."""
    err = _make_rpc_error("decode timeout ERR2008", grpc.StatusCode.INTERNAL)
    assert _is_retryable_grpc(err) is True


# ── SpeechMuxClient.__init__ ──────────────────────────────────────────────────


def test_client_init_creates_insecure_channel():
    """SpeechMuxClient.__init__ must call grpc.insecure_channel with the server address."""
    with (
        patch("speechmux_cli.client.grpc_client.grpc.insecure_channel") as mock_channel_fn,
        patch("speechmux_cli.client.grpc_client.grpc.channel_ready_future") as mock_ready,
    ):
        mock_channel = MagicMock()
        mock_channel_fn.return_value = mock_channel
        mock_future = MagicMock()
        mock_future.result.return_value = None
        mock_ready.return_value = mock_future

        client = SpeechMuxClient("localhost:50051")
        client.close()

    mock_channel_fn.assert_called_once()
    call_args = mock_channel_fn.call_args
    assert call_args[0][0] == "localhost:50051"


def test_client_init_raises_connection_error_on_timeout():
    """A FutureTimeoutError on channel_ready_future must raise ConnectionError."""
    with (
        patch("speechmux_cli.client.grpc_client.grpc.insecure_channel") as mock_channel_fn,
        patch("speechmux_cli.client.grpc_client.grpc.channel_ready_future") as mock_ready,
    ):
        mock_channel = MagicMock()
        mock_channel_fn.return_value = mock_channel
        mock_future = MagicMock()
        mock_future.result.side_effect = grpc.FutureTimeoutError()
        mock_ready.return_value = mock_future

        with pytest.raises(ConnectionError, match="localhost:50051"):
            SpeechMuxClient("localhost:50051", connect_timeout=0.001)


# ── SpeechMuxClient.stream ────────────────────────────────────────────────────


def _make_client_with_mock_stub() -> tuple[SpeechMuxClient, MagicMock]:
    """Construct a SpeechMuxClient whose gRPC channel and stub are mocked.

    Returns:
        Tuple of (client, mock_stub).
    """
    with (
        patch("speechmux_cli.client.grpc_client.grpc.insecure_channel") as mock_channel_fn,
        patch("speechmux_cli.client.grpc_client.grpc.channel_ready_future") as mock_ready,
        patch("speechmux_cli.client.grpc_client.client_pb2_grpc.STTServiceStub") as mock_stub_cls,
    ):
        mock_channel = MagicMock()
        mock_channel_fn.return_value = mock_channel
        mock_future = MagicMock()
        mock_future.result.return_value = None
        mock_ready.return_value = mock_future

        mock_stub = MagicMock()
        mock_stub_cls.return_value = mock_stub

        client = SpeechMuxClient.__new__(SpeechMuxClient)
        client._channel = mock_channel
        client._stub = mock_stub
        client._api_key = ""
        return client, mock_stub


def _make_session_created_response() -> MagicMock:
    """Build a mock StreamingRecognizeResponse carrying a session_created payload.

    Returns:
        Configured MagicMock.
    """
    resp = MagicMock()
    resp.HasField.side_effect = lambda field: field == "session_created"
    return resp


def _make_result_response(
    text: str = "hello",
    committed_text: str = "",
    unstable_text: str = "",
    is_final: bool = True,
) -> MagicMock:
    """Build a mock StreamingResponse carrying a recognition result.

    Args:
        text: Final transcript text.
        committed_text: Committed partial text.
        unstable_text: Unstable partial text.
        is_final: Whether the result is final.

    Returns:
        Configured MagicMock.
    """
    meta = MagicMock()
    meta.latency_sec = 0.1
    meta.real_time_factor = 0.5
    meta.utterance_index = 0

    recognition_result = MagicMock()
    recognition_result.is_final = is_final
    recognition_result.text = text
    recognition_result.committed_text = committed_text
    recognition_result.unstable_text = unstable_text
    recognition_result.audio_duration = 1.0
    recognition_result.language_code = "en"
    recognition_result.meta = meta

    resp = MagicMock()
    resp.HasField.side_effect = lambda field: field == "result"
    resp.result = recognition_result
    return resp


def test_stream_yields_result_after_session_created():
    """stream() must yield a StreamResult for each recognition result response."""
    client, mock_stub = _make_client_with_mock_stub()

    session_resp = _make_session_created_response()
    result_resp = _make_result_response(text="hello world", is_final=True)

    mock_stub.StreamingRecognize.return_value = iter([session_resp, result_resp])

    results = list(client.stream(iter([b"\x00" * 320])))

    assert len(results) == 1
    assert isinstance(results[0], StreamResult)
    assert results[0].text == "hello world"
    assert results[0].is_final is True


def test_stream_yields_partial_result():
    """stream() must correctly populate a partial StreamResult."""
    client, mock_stub = _make_client_with_mock_stub()

    session_resp = _make_session_created_response()
    result_resp = _make_result_response(
        text="",
        committed_text="committed part",
        unstable_text="unstable part",
        is_final=False,
    )

    mock_stub.StreamingRecognize.return_value = iter([session_resp, result_resp])

    results = list(client.stream(iter([])))

    assert len(results) == 1
    assert results[0].is_final is False
    assert results[0].committed_text == "committed part"
    assert results[0].unstable_text == "unstable part"


def test_stream_raises_speech_mux_error_on_rpc_error_at_first_response():
    """grpc.RpcError on the first next() call must raise SpeechMuxError."""
    client, mock_stub = _make_client_with_mock_stub()

    # grpc.RpcError is not a concrete exception; build a minimal real subclass
    # that also mixes in grpc.Call so details()/code() are present.
    class _FakeRpcError(grpc.RpcError, grpc.Call):
        def code(self) -> grpc.StatusCode:
            return grpc.StatusCode.UNAVAILABLE

        def details(self) -> str:
            return "service unavailable"

    rpc_error = _FakeRpcError()

    class _RaisingIter:
        def __iter__(self):
            return self

        def __next__(self):
            raise rpc_error

    mock_stub.StreamingRecognize.return_value = _RaisingIter()

    with pytest.raises(SpeechMuxError):
        list(client.stream(iter([])))


def test_stream_raises_speech_mux_error_on_server_error_response():
    """A StreamingResponse with error field must raise SpeechMuxError."""
    client, mock_stub = _make_client_with_mock_stub()

    session_resp = _make_session_created_response()

    server_error = MagicMock()
    server_error.message = "something went wrong"
    server_error.error_code = "ERR2001"
    server_error.retryable = False

    error_resp = MagicMock()
    error_resp.HasField.side_effect = lambda field: field == "error"
    error_resp.error = server_error

    mock_stub.StreamingRecognize.return_value = iter([session_resp, error_resp])

    with pytest.raises(SpeechMuxError) as exc_info:
        list(client.stream(iter([])))

    assert exc_info.value.code == "ERR2001"


def test_stream_raises_when_first_response_is_not_session_created():
    """An unexpected first response type must raise SpeechMuxError."""
    client, mock_stub = _make_client_with_mock_stub()

    unexpected_resp = MagicMock()
    unexpected_resp.HasField.return_value = False
    unexpected_resp.WhichOneof.return_value = "result"

    mock_stub.StreamingRecognize.return_value = iter([unexpected_resp])

    with pytest.raises(SpeechMuxError, match="Unexpected first response"):
        list(client.stream(iter([])))


def test_stream_raises_when_server_closes_immediately():
    """Server closing without any response must raise SpeechMuxError."""
    client, mock_stub = _make_client_with_mock_stub()

    mock_stub.StreamingRecognize.return_value = iter([])

    with pytest.raises(SpeechMuxError, match="session_created"):
        list(client.stream(iter([])))


# ── engine_hint and vad_mode forwarding ──────────────────────────────────────


def test_stream_engine_hint_forwarded_in_recognition_config():
    """engine_hint must appear in RecognitionConfig of the first request."""
    client, mock_stub = _make_client_with_mock_stub()

    session_resp = _make_session_created_response()
    mock_stub.StreamingRecognize.return_value = iter([session_resp])

    list(client.stream(iter([]), engine_hint="faster-whisper"))

    mock_stub.StreamingRecognize.assert_called_once()
    call_args = mock_stub.StreamingRecognize.call_args
    requests = list(call_args[0][0])  # first positional arg is the request iterator

    first_request = requests[0]
    assert first_request.session_config.recognition_config.engine_hint == "faster-whisper"


def test_stream_vad_mode_auto_end_mapped_to_proto_enum():
    """vad_mode="auto-end" must map to VAD_MODE_AUTO_END in VADConfig."""
    assert _vad_mode_enum("auto-end") == client_pb2.VAD_MODE_AUTO_END


def test_stream_vad_mode_empty_mapped_to_unspecified():
    """vad_mode="" must map to VAD_MODE_UNSPECIFIED in VADConfig."""
    assert _vad_mode_enum("") == client_pb2.VAD_MODE_UNSPECIFIED


def test_stream_result_start_sec_end_sec_populated():
    """StreamResult.start_sec and end_sec must be populated from the proto response."""
    client, mock_stub = _make_client_with_mock_stub()

    session_resp = _make_session_created_response()

    recognition_result = MagicMock()
    recognition_result.is_final = True
    recognition_result.text = "hi"
    recognition_result.committed_text = ""
    recognition_result.unstable_text = ""
    recognition_result.audio_duration = 2.0
    recognition_result.language_code = "en"
    recognition_result.start_sec = 1.5
    recognition_result.end_sec = 3.2
    recognition_result.meta = None

    result_resp = MagicMock()
    result_resp.HasField.side_effect = lambda f: f == "result"
    result_resp.result = recognition_result

    mock_stub.StreamingRecognize.return_value = iter([session_resp, result_resp])

    results = list(client.stream(iter([])))

    assert len(results) == 1
    assert results[0].start_sec == pytest.approx(1.5)
    assert results[0].end_sec == pytest.approx(3.2)
