"""Audio file loading, resampling, and chunking for SpeechMux."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterator
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly  # type: ignore[import-untyped]

TARGET_SAMPLE_RATE = 16000
DEFAULT_CHUNK_MS = 100
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB


class AudioLoadError(Exception):
    """Raised when an audio file cannot be loaded or validated."""


def _ffmpeg_probe(path: Path) -> float:
    """Return audio duration in seconds by probing with ffmpeg.

    Args:
        path: Path to the audio file.

    Returns:
        Duration in seconds.

    Raises:
        AudioLoadError: If ffmpeg is not found or cannot parse the file.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(path)],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise AudioLoadError(
            f"ffmpeg not found — install ffmpeg to support this format: {path}"
        ) from error
    # ffmpeg writes stream info to stderr even when no output is given.
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", result.stderr)
    if not match:
        raise AudioLoadError(
            f"Cannot read audio file (unsupported format or not audio): {path}"
        )
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _ffmpeg_chunks(path: Path, chunk_ms: int) -> Iterator[bytes]:
    """Decode audio file to PCM S16LE chunks via ffmpeg.

    Decodes directly to 16 kHz mono S16LE, bypassing soundfile and scipy.

    Args:
        path: Path to the audio file.
        chunk_ms: Target chunk duration in milliseconds.

    Yields:
        PCM S16LE bytes of approximately chunk_ms milliseconds each.

    Raises:
        AudioLoadError: If ffmpeg is not found or exits with an error.
    """
    bytes_per_chunk = TARGET_SAMPLE_RATE * 2 * chunk_ms // 1000  # int16 = 2 bytes
    try:
        proc = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", str(path),
                "-f", "s16le", "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE),
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as error:
        raise AudioLoadError(
            f"ffmpeg not found — install ffmpeg to support this format: {path}"
        ) from error

    assert proc.stdout is not None
    buffer = b""
    try:
        while True:
            chunk = proc.stdout.read(bytes_per_chunk)
            if not chunk:
                break
            buffer += chunk
            while len(buffer) >= bytes_per_chunk:
                yield buffer[:bytes_per_chunk]
                buffer = buffer[bytes_per_chunk:]
    finally:
        proc.stdout.close()
        proc.wait()

    if buffer:
        yield buffer

    if proc.returncode != 0:
        stderr_output = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        raise AudioLoadError(
            f"ffmpeg decode failed for {path}: {stderr_output.strip()}"
        )


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample audio from source_rate to target_rate using scipy polyphase filter.

    Args:
        audio: 1-D float32 audio array.
        source_rate: Original sample rate in Hz.
        target_rate: Target sample rate in Hz.

    Returns:
        Resampled float32 audio array.
    """
    if source_rate == target_rate:
        return audio
    common_divisor = gcd(source_rate, target_rate)
    return resample_poly(
        audio, target_rate // common_divisor, source_rate // common_divisor
    ).astype(np.float32)


def _to_pcm16(audio: np.ndarray) -> bytes:
    """Convert float32 audio [-1, 1] to PCM S16LE bytes.

    Args:
        audio: 1-D float32 audio array with values in [-1, 1].

    Returns:
        Raw PCM S16LE bytes.
    """
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


class AudioLoader:
    """Loads an audio file and yields PCM S16LE chunks ready for SpeechMux."""

    def __init__(
        self,
        path: str | Path,
        *,
        chunk_ms: int = DEFAULT_CHUNK_MS,
        max_file_size: int = MAX_FILE_SIZE_BYTES,
    ) -> None:
        """Initialize AudioLoader.

        Args:
            path: Path to the audio file.
            chunk_ms: Target chunk duration in milliseconds.
            max_file_size: Maximum allowed file size in bytes.
        """
        self._path = Path(path)
        self._chunk_ms = chunk_ms
        self._max_file_size = max_file_size
        self._sample_rate: int | None = None
        self._duration_sec: float | None = None
        self._use_ffmpeg: bool = False
        self._validated: bool = False

    def validate(self) -> None:
        """Validate file existence, size, and format.

        Populates _sample_rate and _duration_sec as a side effect.

        Raises:
            AudioLoadError: If the file is missing, empty, too large, or unreadable.
        """
        if self._validated:
            return
        if not self._path.exists():
            raise AudioLoadError(f"File not found: {self._path}")
        if not self._path.is_file():
            raise AudioLoadError(f"Not a file: {self._path}")

        file_size = self._path.stat().st_size
        if file_size == 0:
            raise AudioLoadError(f"File is empty: {self._path}")
        max_size_mb = self._max_file_size // (1024 * 1024)
        if file_size > self._max_file_size:
            raise AudioLoadError(
                f"File too large ({file_size // (1024 * 1024)}MB), "
                f"limit is {max_size_mb}MB: {self._path}"
            )

        try:
            audio_info = sf.info(str(self._path))
            self._sample_rate = audio_info.samplerate
            self._duration_sec = audio_info.duration
            self._use_ffmpeg = False
        except Exception:
            # soundfile cannot handle this container (e.g. WebM, MP4).
            # Fall back to ffmpeg for probe and decode.
            self._duration_sec = _ffmpeg_probe(self._path)
            self._sample_rate = TARGET_SAMPLE_RATE
            self._use_ffmpeg = True
        self._validated = True

    @property
    def duration_sec(self) -> float:
        """Audio duration in seconds.

        Returns:
            Duration in seconds, or 0.0 if the file has not been validated yet.
        """
        if self._duration_sec is None:
            self.validate()
        return self._duration_sec or 0.0

    @property
    def sample_rate(self) -> int:
        """Sample rate of the source audio file in Hz.

        Returns:
            Source sample rate, or TARGET_SAMPLE_RATE if not yet validated.
        """
        if self._sample_rate is None:
            self.validate()
        return self._sample_rate or TARGET_SAMPLE_RATE

    def chunks(self) -> Iterator[bytes]:
        """Yield PCM S16LE chunks at TARGET_SAMPLE_RATE (16kHz), mono.

        Reads the file in blocks to avoid loading the entire file into memory.
        Resamples to 16kHz if the source rate differs.

        Yields:
            PCM S16LE bytes of approximately chunk_ms milliseconds each.

        Raises:
            AudioLoadError: If the file cannot be read.
        """
        self.validate()

        if self._use_ffmpeg:
            yield from _ffmpeg_chunks(self._path, self._chunk_ms)
            return

        samples_per_chunk = max(1, TARGET_SAMPLE_RATE * self._chunk_ms // 1000)
        source_rate = self._sample_rate or TARGET_SAMPLE_RATE

        # Block size in source samples (will be resampled to 16kHz if needed).
        if source_rate == TARGET_SAMPLE_RATE:
            block_size = samples_per_chunk
        else:
            # Read larger blocks to amortise resample overhead (~250ms blocks).
            block_size = max(samples_per_chunk, source_rate // 4)

        sample_buffer = np.array([], dtype=np.float32)

        try:
            with sf.SoundFile(str(self._path)) as audio_file:
                for block in audio_file.blocks(
                    blocksize=block_size, dtype="float32", always_2d=True
                ):
                    mono_block = block[:, 0] if block.ndim > 1 else block.ravel()
                    resampled_block = _resample(mono_block, source_rate, TARGET_SAMPLE_RATE)
                    sample_buffer = np.concatenate([sample_buffer, resampled_block])

                    # Emit complete chunks.
                    while len(sample_buffer) >= samples_per_chunk:
                        yield _to_pcm16(sample_buffer[:samples_per_chunk])
                        sample_buffer = sample_buffer[samples_per_chunk:]

        except sf.SoundFileError as soundfile_error:
            raise AudioLoadError(
                f"Error reading {self._path}: {soundfile_error}"
            ) from soundfile_error

        # Flush remainder.
        if len(sample_buffer) > 0:
            yield _to_pcm16(sample_buffer)
