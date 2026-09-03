#!/usr/bin/env python3
"""
Offline voice command listener (Vosk).

Streams recognized words, one JSON object per line on stdout:
    {"text": "shutdown now"}   # final utterance
    {"text": "shut"}           # partial (unstable) hypothesis, key "partial"
Raw words are NOT interpreted here — the Rust controller decides what they mean
(matching is done there, lowercase and punctuation-free).

Audio device selection:
    --list-devices   show capture devices and exit
    --device N       sounddevice device index (default: system default)
"""
import argparse
import json
import os
import sys

import sounddevice as sd
from vosk import Model, KaldiRecognizer, SetLogLevel

MODEL_DIRS = [
    "vosk-model-small-en-us",
    "vosk-model-small-en-us-0.15",
]
SAMPLE_RATE = 16000
BLOCK_MS = 100  # recognizer feed granularity
BLOCK = SAMPLE_RATE * BLOCK_MS // 1000


def open_stream(device, callback):
    """Open the mic at 16 kHz mono. Raw ALSA hw devices often reject 16 kHz
    (paInvalidSampleRate) while virtual servers (default/pulse/pipewire/jack)
    resample; on failure retry once with the system default."""
    try:
        return sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=BLOCK,
            device=device,
            callback=callback,
        )
    except sd.PortAudioError:
        if device is None:
            raise
        print("requested device failed, falling back to system default", file=sys.stderr)
        return sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=BLOCK,
            device=None,
            callback=callback,
        )


def find_model(script_dir: str) -> str:
    for name in MODEL_DIRS:
        path = os.path.join(script_dir, "models", name)
        if os.path.isdir(path):
            return path
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        print(json.dumps(sd.query_devices(), indent=2))
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = find_model(script_dir)
    if not model_path:
        print("ERROR: vosk model not found under models/", file=sys.stderr)
        sys.exit(1)

    SetLogLevel(-1)  # silence Kaldi chatter; errors still go to stderr
    model = Model(model_path)
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)

    def callback(indata, frames, time, status):
        if status:
            print(f"audio status: {status}", file=sys.stderr)
        if recognizer.AcceptWaveform(bytes(indata)):
            result = json.loads(recognizer.Result())
            text = result.get("text", "").strip()
            if text:
                print(json.dumps({"text": text, "partial": False}), flush=True)
        else:
            result = json.loads(recognizer.PartialResult())
            text = result.get("partial", "").strip()
            if text:
                print(json.dumps({"text": text, "partial": True}), flush=True)

    try:
        with open_stream(args.device, callback):
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
