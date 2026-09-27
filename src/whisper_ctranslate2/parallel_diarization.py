"""Run pyannote diarization in a child process while this process transcribes.

The sequential path (``get_diarization`` before the transcription loop) is kept
as a kill switch: ``WC2_DIARIZATION_MODE=sequential`` in the environment, or a
file ``<venv>/wc2-diarization-mode`` containing ``sequential``.

Nothing here imports torch or pyannote at module level: the CLI process must not
pay for (or be changed by) pyannote just because this module exists.
"""

import atexit
import datetime
import multiprocessing
import os
import signal
import sys
import threading
import time
import traceback

MODE_ENV = "WC2_DIARIZATION_MODE"
MODE_FILE_NAME = "wc2-diarization-mode"
PARALLEL = "parallel"
SEQUENTIAL = "sequential"

_OK = "ok"
_FAILED = "failed"
_PARENT_POLL_SECONDS = 1.0
_REAP_TIMEOUT_SECONDS = 30.0


def diarization_mode():
    """``sequential`` only when explicitly asked for; anything else runs in parallel.

    The environment variable wins over the file, so a single run can be forced
    either way. The file is read on every run: creating or deleting it takes
    effect on the next job without touching the queue workers.
    """
    value = os.environ.get(MODE_ENV)
    if value is None or not value.strip():
        try:
            with open(
                os.path.join(sys.prefix, MODE_FILE_NAME), encoding="utf-8"
            ) as handle:
                value = handle.read()
        except OSError:
            value = ""
    return SEQUENTIAL if value.strip().lower() == SEQUENTIAL else PARALLEL


def mirror_sequential_parent_state(diarization_device, threads):
    """Leave torch in this process as the sequential path leaves it before Whisper runs.

    ctranslate2 4.4.0 already imports torch at startup. Sequentially,
    Diarization() then turns TF32 on, pyannote's fix_reproducibility() turns it
    off again on CUDA, and ``--threads`` caps torch's intra-op pool. CTranslate2
    and faster-whisper 1.0.3's numpy feature extractor ignore all of this; it
    only matters if the Whisper stack ever computes on torch, and then it keeps
    the diarized path's numerics.
    """
    try:
        import torch
    except Exception:
        return  # the child reports the real import failure

    on_cuda = torch.device(diarization_device).type == "cuda"
    torch.backends.cuda.matmul.allow_tf32 = not on_cuda
    torch.backends.cudnn.allow_tf32 = not on_cuda
    if threads > 0:
        torch.set_num_threads(threads)


def is_out_of_memory(error):
    """True for allocation failures, the one class of error parallelism can cause.

    Two CUDA contexts share the card while pyannote runs. Covers torch's
    ``CUDA out of memory`` / ``MPS backend out of memory``, CTranslate2's
    ``CUDA failed with error out of memory``, cuBLAS/cuDNN ``*_ALLOC_FAILED``
    and host ``MemoryError``.
    """
    if isinstance(error, MemoryError):
        return True
    text = str(error).lower()
    return "out of memory" in text or "alloc_failed" in text


class ParallelDiarization:
    """Parent side: start the child, collect its turns, retry or fail the run.

    Parallelism must never fail a job the sequential path would have finished.
    A failure that can come from sharing the machine (out of memory, or the
    child dying without a report: OOM killer, segfault) is retried once in a
    fresh child when the result is first needed, i.e. after transcription,
    which is how the sequential path diarizes. Every other failure (missing
    config, broken install) would fail sequentially too: it ends the run at
    once with exit code 1 and the child's traceback as the last stderr lines.
    """

    def __init__(self, audio, config_path, device, threads, verbose, child_target=None):
        self._child_target = child_target or diarize_in_child  # tests inject fakes
        self._audio = list(audio)
        self._config_path = config_path
        self._device = device
        self._threads = threads
        self._verbose = verbose
        self._turns = None
        self._failure = None
        self._done = threading.Event()
        self._closing = False
        self._waited = False
        self._process = None
        self._receiver = None

    def start(self):
        """Must run on the main thread: PR_SET_PDEATHSIG follows the creating thread."""
        self._process, self._receiver = self._spawn()
        atexit.register(self._mark_closing)
        threading.Thread(
            target=self._watch, name="wc2-diarization-watch", daemon=True
        ).start()

    def turns_for(self, audio_path):
        """Block until every file is diarized; never returns on failure."""
        return self._result()[audio_path]

    def release(self):
        """Wait for diarization to finish and its process to exit, freeing its memory."""
        self._result()
        self._reap()

    def finish(self):
        """A diarization failure fails the run even if every transcription failed."""
        self._result()
        self._reap()

    def _result(self):
        if not self._waited:
            self._waited = True
            started = datetime.datetime.now()
            self._done.wait()
            if self._verbose:
                print(
                    f"Time waited for diarization: {datetime.datetime.now() - started}"
                )
            if self._turns is None:
                self._retry()
        return self._turns

    def _retry(self):
        """Main thread only (see start()); a second failure of any kind ends the run."""
        sys.stderr.write(
            f"Warning: diarization failed ({self._failure}); "
            "retrying once in a fresh process\n"
        )
        _flush_std_streams()
        self._reap()
        self._process, receiver = self._spawn()
        message = _receive(receiver)
        if message is not None and message[0] == _OK:
            self._turns = message[1]
            return
        self._fail(message)

    def _spawn(self):
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=self._child_target,
            args=(
                sender,
                os.getpid(),
                self._audio,
                self._config_path,
                self._device,
                self._threads,
                self._verbose,
            ),
            name="wc2-diarization",
            daemon=True,
        )
        process.start()
        sender.close()  # the child now holds the only write end: its exit reads as EOF
        return process, receiver

    def _reap(self):
        self._process.join(_REAP_TIMEOUT_SECONDS)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(5)

    def _mark_closing(self):
        self._closing = True

    def _watch(self):
        message = _receive(self._receiver)
        if message is not None and message[0] == _OK:
            self._turns = message[1]
        elif self._closing:
            return  # the CLI is already exiting for its own reason
        elif message is not None and not message[2]:
            self._fail(message)  # would fail sequentially too: fail fast
        elif message is not None:
            self._failure = message[1]
        else:
            self._process.join(5)
            self._failure = f"process exited with code {self._process.exitcode}"
        self._done.set()

    def _fail(self, message):
        if message is None:
            self._process.join(5)
            sys.stderr.write(
                "Error: diarization process exited with code "
                f"{self._process.exitcode} before returning a result\n"
            )
        # On _FAILED the child has already printed its traceback to the shared
        # stderr; its last line is the exception, exactly as the sequential
        # path's uncaught exception printed it.
        _flush_std_streams()
        os._exit(1)


def diarize_in_child(sender, parent_pid, audio, config_path, device, threads, verbose):
    """Child side: the sequential path's own diarization code, in a fresh process."""
    _die_with_parent(parent_pid)
    try:
        from .diarization import Diarization
        from .speaker_assignment import annotation_to_turns
        from .whisper_ctranslate2 import get_diarization

        diarize_model = Diarization(use_auth_token=config_path, device=device)
        if threads > 0:
            diarize_model.set_threads(threads)
        diarization_output = get_diarization(audio, diarize_model, verbose)
        turns = {
            audio_path: annotation_to_turns(annotation)
            for audio_path, annotation in diarization_output.items()
        }
    except BaseException as error:
        traceback.print_exc()
        _flush_std_streams()
        sender.send(
            (_FAILED, f"{type(error).__name__}: {error}", is_out_of_memory(error))
        )
        sender.close()
        return

    _flush_std_streams()
    sender.send((_OK, turns))
    sender.close()


def _receive(receiver):
    """The child's one message, or None if it exited without sending one."""
    try:
        return receiver.recv()
    except Exception:
        return None


def _die_with_parent(parent_pid):
    """Never outlive the CLI: a Process timeout SIGKILLs it and nothing else."""
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            pr_set_pdeathsig = 1
            if libc.prctl(pr_set_pdeathsig, signal.SIGKILL, 0, 0, 0) != 0:
                raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
        except Exception as error:
            sys.stderr.write(f"Warning: parent-death signal not set: {error}\n")

    # The parent may have died before prctl() ran.
    if os.getppid() != parent_pid:
        os._exit(1)

    threading.Thread(
        target=_watch_parent,
        args=(parent_pid,),
        name="wc2-parent-watchdog",
        daemon=True,
    ).start()


def _watch_parent(parent_pid):
    """Fallback where prctl is unavailable (macOS) or failed."""
    while True:
        time.sleep(_PARENT_POLL_SECONDS)
        if os.getppid() != parent_pid:
            os._exit(1)  # ends every thread at once, like the SIGKILL above


def _flush_std_streams():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
