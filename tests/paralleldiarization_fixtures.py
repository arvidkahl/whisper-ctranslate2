"""Child targets and parent drivers for testparalleldiarization.py.

They live in their own importable module because spawn children import the
target by reference; a function defined in a test's ``__main__`` is not
importable from the child.

Targets that must behave differently on a retry keep their state in a file:
``config_path`` doubles as that file's path (it is otherwise unused by fakes).
"""

import functools
import os
import signal
import sys
import time
from unittest import mock

from src.whisper_ctranslate2 import parallel_diarization


def _send_turns(sender, audio):
    sender.send(
        (
            parallel_diarization._OK,
            {path: [(0.0, 1.5, "A", "SPEAKER_00")] for path in audio},
        )
    )
    sender.close()


def _attempt(state_path):
    """1 on the first call for this state file, 2 on the second, ..."""
    attempt = 1
    if os.path.exists(state_path):
        with open(state_path) as handle:
            attempt = int(handle.read()) + 1
    with open(state_path, "w") as handle:
        handle.write(str(attempt))
    return attempt


def _report_failure(sender, line, retryable):
    sys.stderr.write("Traceback (most recent call last):\n  (fake)\n" + line + "\n")
    sys.stderr.flush()
    sender.send((parallel_diarization._FAILED, line, retryable))
    sender.close()


def child_sends_turns(sender, parent_pid, audio, config_path, device, threads, verbose):
    time.sleep(0.5)
    _send_turns(sender, audio)


def child_sends_many_turns(
    sender, parent_pid, audio, config_path, device, threads, verbose
):
    # ~200k turns (several MB pickled): far beyond the pipe buffer, like a
    # multi-hour episode would never even reach.
    turns = [
        (i * 0.5, i * 0.5 + 0.4, f"T{i}", f"SPEAKER_{i % 12:02d}")
        for i in range(200_000)
    ]
    sender.send((parallel_diarization._OK, {path: turns for path in audio}))
    sender.close()


def child_dies_silently(
    sender, parent_pid, audio, config_path, device, threads, verbose
):
    os._exit(3)


def child_dies_once_then_succeeds(
    sender, parent_pid, audio, config_path, device, threads, verbose
):
    if _attempt(config_path) == 1:
        os.kill(os.getpid(), signal.SIGKILL)  # like the OOM killer: no report at all
    _send_turns(sender, audio)


def child_out_of_memory_once(
    sender, parent_pid, audio, config_path, device, threads, verbose
):
    if _attempt(config_path) == 1:
        _report_failure(sender, "OutOfMemoryError: CUDA out of memory.", True)
        return
    _send_turns(sender, audio)


def child_out_of_memory_always(
    sender, parent_pid, audio, config_path, device, threads, verbose
):
    attempt = _attempt(config_path)
    _report_failure(
        sender, f"OutOfMemoryError: CUDA out of memory (attempt {attempt}).", True
    )


def child_fails_for_good(
    sender, parent_pid, audio, config_path, device, threads, verbose
):
    _attempt(config_path)
    _report_failure(sender, "ValueError: broken pipeline config", False)


def child_sleeps_forever(
    sender, parent_pid, audio, config_path, device, threads, verbose
):
    parallel_diarization._die_with_parent(parent_pid)
    with open(config_path, "w") as handle:  # config_path doubles as the pid file
        handle.write(str(os.getpid()))
    while True:
        time.sleep(1)


class FakeTranscribe:
    """Stands in for Transcribe in main(); ``failures`` are raised by inference() in order."""

    failures = []

    def __init__(self, *args, **kwargs):
        pass

    def inference(self, audio, task, language, verbose, live, options):
        if FakeTranscribe.failures:
            raise FakeTranscribe.failures.pop(0)
        print("fake transcription ran", flush=True)
        return {
            "text": " hello",
            "language": "en",
            "segments": [
                {"id": 0, "seek": 0, "start": 0.0, "end": 1.0, "text": " hello"}
            ],
        }


def run_cli(output_dir, whisper_failures):
    """main() with a fake Whisper and a fake diarization child."""
    from src.whisper_ctranslate2 import whisper_ctranslate2 as cli

    FakeTranscribe.failures = list(whisper_failures)
    parallel = functools.partial(
        parallel_diarization.ParallelDiarization, child_target=child_sends_turns
    )
    real_release = parallel_diarization.ParallelDiarization.release

    def release(self):
        real_release(self)
        print(f"released; child alive: {self._process.is_alive()}", flush=True)

    argv = [
        "whisper-ctranslate2",
        "--hf_token",
        "/unused/pyannote_diarization_config.yaml",
        "--output_dir",
        output_dir,
        "--output_format",
        "srt",
        "--language",
        "en",
        "/nonexistent/episode.wav",
    ]
    with mock.patch.object(cli, "Transcribe", FakeTranscribe), mock.patch.object(
        cli, "ParallelDiarization", parallel
    ), mock.patch.object(
        parallel_diarization.ParallelDiarization, "release", release
    ), mock.patch.object(
        sys, "argv", argv
    ), mock.patch.dict(
        os.environ, {parallel_diarization.MODE_ENV: "parallel"}
    ):
        cli.main()


if __name__ == "__main__":
    # Parent process for the subprocess-level tests:
    #   python paralleldiarization_fixtures.py <target> <config_path> [release]
    #   python paralleldiarization_fixtures.py cli <output_dir> <whisper failure>...
    if sys.argv[1] == "cli":
        run_cli(sys.argv[2], [RuntimeError(text) for text in sys.argv[3:]])
        print("finished normally", flush=True)
        sys.exit(0)

    target_name, config_path = sys.argv[1], sys.argv[2]
    target = (
        parallel_diarization.diarize_in_child
        if target_name == "real"
        else globals()[target_name]
    )
    runner = parallel_diarization.ParallelDiarization(
        ["/nonexistent/audio.wav"], config_path, "cpu", 0, True, child_target=target
    )
    runner.start()
    if sys.argv[3:] == ["release"]:
        runner.release()
        print(f"released; child alive: {runner._process.is_alive()}", flush=True)
    runner.finish()
    print("finished normally", flush=True)
