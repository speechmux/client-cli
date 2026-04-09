# speechmux/client-cli

Command-line client for SpeechMux. Streams audio to Core via gRPC and displays transcription results in real time. Supports file transcription, batch processing, and live microphone input.

## Structure

```
client-cli/
├── speechmux_cli/
│   ├── __main__.py           # click CLI group + global options
│   ├── types.py              # StreamResult, SpeechMuxError
│   ├── client/
│   │   └── grpc_client.py    # gRPC StreamingRecognize client
│   ├── audio/
│   │   └── loader.py         # soundfile block reader + resampling
│   └── commands/
│       ├── _output.py        # Terminal display (partial/final text)
│       ├── file.py           # speechmux file <audio>
│       ├── batch.py          # speechmux batch <directory>
│       └── mic.py            # speechmux mic
├── tests/
│   └── test_grpc_client.py   # 25 tests (StreamResult, gRPC mocking)
└── pyproject.toml
```

## Install

```bash
# Base (file + batch commands)
pip install -e "."

# With microphone support
pip install -e ".[mic]"
```

## Usage

### Global Options

```
speechmux [OPTIONS] COMMAND [ARGS]

Options:
  --server TEXT              Core gRPC address (default: localhost:50051)
  --api-key TEXT             API key (env: SPEECHMUX_API_KEY)
  --lang TEXT                BCP-47 language code (empty = auto-detect)
  --task [transcribe|translate]  Task type (default: transcribe)
  --profile [realtime|accurate]  Decode profile (default: realtime)
  --vad-silence FLOAT        VAD silence duration in seconds (default: 0.8)
  --vad-threshold FLOAT      VAD speech probability threshold (default: 0.5)
  --connect-timeout FLOAT    Connection timeout in seconds (default: 10.0)
  --session-timeout FLOAT    Max session duration in seconds (default: 300.0)
  --tls                      Enable TLS
  --tls-ca-file TEXT         Path to CA certificate for TLS
```

### File Transcription

Transcribe a single audio file. Supports WAV, FLAC, OGG, MP3, and other formats via `soundfile`.

```bash
# Basic transcription
speechmux file audio.wav --lang ko

# With metrics output (latency, RTF)
speechmux file audio.wav --lang ko --metrics

# Translate to English
speechmux file audio.wav --task translate

# High-accuracy decode profile
speechmux file recording.wav --lang en --profile accurate
```

Options:
- `--chunk-ms INT` — audio chunk size in milliseconds (default: 100)
- `--realtime` — pace audio delivery at real-time speed
- `--metrics` — print latency and real-time factor after transcription
- `--json` — output results as JSON lines

### Batch Processing

Transcribe all audio files in a directory. Uses `ThreadPoolExecutor` for parallel processing.

```bash
# Basic batch
speechmux batch ./audio_dir/ --lang ko --output ./results/

# Parallel workers
speechmux batch ./audio_dir/ --lang ko --workers 4

# JSON output
speechmux batch ./audio_dir/ --lang ko --json --output ./results/

# Resume interrupted batch (skip already-processed files)
speechmux batch ./audio_dir/ --lang ko --output ./results/ --resume

# Continue on errors
speechmux batch ./audio_dir/ --lang ko --on-error continue
```

Options:
- `-o, --output DIR` — output directory for per-file JSON results
- `--workers INT` — number of parallel workers (default: 4)
- `--chunk-ms INT` — audio chunk size in milliseconds (default: 100)
- `--on-error [continue|stop]` — error handling strategy (default: continue)
- `--resume` — skip files that already have results
- `--json` — output results as JSON lines

After completion, prints a summary: success/skip/fail counts.

### Live Microphone

Stream audio from the default microphone in real time. Requires the `mic` extra (`sounddevice`).

```bash
# Default language (auto-detect)
speechmux mic

# Korean
speechmux mic --lang ko

# Specific input device
speechmux mic --device "MacBook Pro Microphone"

# Stop with Ctrl+C
```

The terminal displays partial (unstable) text as it streams, replacing it with final (committed) text when each utterance completes.

Options:
- `--device TEXT` — input device name or index (default: system default)
- `--json` — output results as JSON lines

## gRPC Client

`grpc_client.py` implements the `StreamingRecognize` bidirectional streaming protocol:

1. Opens a gRPC channel to Core (with optional TLS)
2. Sends `SessionConfig` as the first message (language, task, profile, VAD settings)
3. Receives `SessionCreated` with negotiated settings
4. Streams audio chunks and receives `RecognitionResult` messages
5. Sends `StreamSignal { is_last: true }` when audio ends
6. Computes dynamic timeout per file: `audio_duration × 3 + 30s`

## Test

```bash
python -m pytest tests/ -v
```

## License

MIT
