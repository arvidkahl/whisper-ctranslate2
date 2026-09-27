import copy
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from collections import OrderedDict
from unittest import mock

import importlib.util

import numpy as np

try:
    from pyannote.core import Annotation, Segment
except ImportError:  # the fork's CI installs neither pyannote nor torch
    Annotation = Segment = None

from src.whisper_ctranslate2 import parallel_diarization
from src.whisper_ctranslate2.speaker_assignment import (
    annotation_to_turns,
    assign_speakers,
    assign_speakers_from_turns,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAS_PYANNOTE_AUDIO = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "pyannote.audio")
)
FIXTURES = os.path.join(REPO_ROOT, "tests", "paralleldiarization_fixtures.py")


def reference_assign(diarize_data, transcript_result, speaker_name):
    """Diarization._do_assign_speakers_to_segments verbatim at 0186e49."""
    diarize_df = np.array(
        diarize_data,
        dtype=[("segment", object), ("label", object), ("speaker", object)],
    )
    diarize_df = np.core.records.fromarrays(
        [
            diarize_df["segment"],
            diarize_df["label"],
            diarize_df["speaker"],
            np.array([seg.start for seg in diarize_df["segment"]]),
            np.array([seg.end for seg in diarize_df["segment"]]),
            np.zeros(len(diarize_df)),
        ],
        names="segment, label, speaker, start, end, intersection",
    )
    for seg in transcript_result["segments"]:
        intersection = np.minimum(diarize_df["end"], seg["end"]) - np.maximum(
            diarize_df["start"], seg["start"]
        )
        diarize_df["intersection"] = intersection
        dia_segment = diarize_df[diarize_df["intersection"] > 0]
        if len(dia_segment) > 0:
            speakers = {}
            for item in dia_segment:
                speaker = item["speaker"]
                old_i = speakers.get(speaker, 0)
                speakers[speaker] = old_i + item["intersection"]
            sorted_dict = OrderedDict(
                sorted(speakers.items(), key=lambda x: x[1], reverse=True)
            )
            first_item = next(iter(sorted_dict.items()))
            if first_item:
                speaker = first_item[0]
                if speaker_name:
                    speaker = speaker.replace("SPEAKER", speaker_name)
                seg["speaker"] = speaker
    return transcript_result


def fixture_annotation():
    """Overlaps, a tie, back-to-back turns, one speaker on two tracks, float noise."""
    annotation = Annotation()
    annotation[Segment(0.03096875, 4.21596875), "A"] = "SPEAKER_00"
    annotation[Segment(3.9, 9.412), "B"] = "SPEAKER_01"
    annotation[Segment(9.412, 12.0), "C"] = "SPEAKER_00"
    annotation[Segment(10.5, 11.5), "D"] = "SPEAKER_02"
    annotation[Segment(20.1, 21.1), "E"] = "SPEAKER_01"
    annotation[Segment(21.1, 22.1), "F"] = "SPEAKER_02"
    annotation[Segment(30.0000001, 33.3333333), "G"] = "SPEAKER_00"
    return annotation


def fixture_transcript():
    return {
        "text": "irrelevant",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 2.0, "text": " a"},
            {"start": 3.5, "end": 5.0, "text": " b"},
            {"start": 9.0, "end": 10.9, "text": " c"},
            {"start": 10.6, "end": 11.4, "text": " d"},
            {"start": 15.0, "end": 16.0, "text": " no speaker"},
            {"start": 20.1, "end": 22.1, "text": " tie"},
            {"start": 29.0, "end": 40.0, "text": " e"},
        ],
    }


@unittest.skipIf(Annotation is None, "needs pyannote.core")
class TestSpeakerAssignmentEquivalence(unittest.TestCase):
    def test_turns_path_matches_the_pre_change_implementation(self):
        annotation = fixture_annotation()
        for speaker_name in (None, "", "SPEAKER", "PARLANT"):
            with self.subTest(speaker_name=speaker_name):
                expected = reference_assign(
                    list(annotation.itertracks(yield_label=True)),
                    fixture_transcript(),
                    speaker_name,
                )
                turns = pickle.loads(pickle.dumps(annotation_to_turns(annotation)))
                actual = assign_speakers_from_turns(
                    turns, fixture_transcript(), speaker_name
                )
                self.assertEqual(expected, actual)

    def test_moved_function_matches_the_pre_change_implementation(self):
        annotation = fixture_annotation()
        diarize_data = list(annotation.itertracks(yield_label=True))
        self.assertEqual(
            reference_assign(diarize_data, fixture_transcript(), None),
            assign_speakers(diarize_data, fixture_transcript(), None),
        )

    def test_turns_survive_pickling_bit_for_bit(self):
        turns = annotation_to_turns(fixture_annotation())
        self.assertEqual(turns, pickle.loads(pickle.dumps(turns)))
        self.assertEqual(
            [(t[0].hex(), t[1].hex()) for t in turns],
            [(t[0].hex(), t[1].hex()) for t in pickle.loads(pickle.dumps(turns))],
        )

    def test_zero_turns_leave_segments_unlabelled_in_both_paths(self):
        expected = reference_assign([], fixture_transcript(), None)
        actual = assign_speakers_from_turns([], fixture_transcript(), None)
        self.assertEqual(expected, actual)
        self.assertFalse(any("speaker" in s for s in actual["segments"]))

    def test_tie_is_broken_the_same_way(self):
        annotation = fixture_annotation()
        expected = reference_assign(
            list(annotation.itertracks(yield_label=True)), fixture_transcript(), None
        )
        actual = assign_speakers_from_turns(
            annotation_to_turns(annotation), copy.deepcopy(fixture_transcript()), None
        )
        self.assertEqual(expected["segments"][5], actual["segments"][5])


class TestDiarizationMode(unittest.TestCase):
    def setUp(self):
        self.prefix = tempfile.mkdtemp()
        patcher = mock.patch.object(sys, "prefix", self.prefix)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _write_mode_file(self, content):
        with open(
            os.path.join(self.prefix, parallel_diarization.MODE_FILE_NAME), "w"
        ) as handle:
            handle.write(content)

    def test_parallel_is_the_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual("parallel", parallel_diarization.diarization_mode())

    def test_environment_selects_sequential(self):
        for value in ("sequential", " Sequential\n", "SEQUENTIAL"):
            with mock.patch.dict(os.environ, {"WC2_DIARIZATION_MODE": value}):
                self.assertEqual("sequential", parallel_diarization.diarization_mode())

    def test_unknown_values_stay_parallel(self):
        for value in ("parallel", "off", "0", "seq"):
            with mock.patch.dict(os.environ, {"WC2_DIARIZATION_MODE": value}):
                self.assertEqual("parallel", parallel_diarization.diarization_mode())

    def test_file_in_the_venv_selects_sequential(self):
        self._write_mode_file("sequential\n")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual("sequential", parallel_diarization.diarization_mode())

    def test_environment_wins_over_the_file(self):
        self._write_mode_file("sequential\n")
        with mock.patch.dict(os.environ, {"WC2_DIARIZATION_MODE": "parallel"}):
            self.assertEqual("parallel", parallel_diarization.diarization_mode())


class TestOutOfMemoryClassification(unittest.TestCase):
    def test_allocation_failures_are_out_of_memory(self):
        for error in (
            RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB"),
            RuntimeError("CUDA failed with error out of memory"),
            RuntimeError("MPS backend out of memory (MPS allocated: 1 GB)"),
            RuntimeError("cuBLAS failed with status CUBLAS_STATUS_ALLOC_FAILED"),
            RuntimeError("cuDNN error: CUDNN_STATUS_ALLOC_FAILED"),
            MemoryError(),
        ):
            with self.subTest(error=repr(error)):
                self.assertTrue(parallel_diarization.is_out_of_memory(error))

    def test_other_failures_are_not(self):
        for error in (
            FileNotFoundError("Config file not found: /x/models/config.yaml"),
            NameError("name 'Pipeline' is not defined"),
            RuntimeError("CUDA failed with error an illegal memory access"),
            ValueError("bad"),
        ):
            with self.subTest(error=repr(error)):
                self.assertFalse(parallel_diarization.is_out_of_memory(error))


@unittest.skipIf(sys.platform == "win32", "POSIX process semantics")
class TestChildProcessLifecycle(unittest.TestCase):
    """Real processes: spawn, EOF, exit codes, retries, orphan safety."""

    def _run_parent(self, target, config_path, *extra, timeout=120):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [REPO_ROOT] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
        )
        return subprocess.run(
            [sys.executable, FIXTURES, target, config_path, *extra],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=tempfile.gettempdir(),
        )

    def test_turns_are_delivered(self):
        result = self._run_parent("child_sends_turns", "/unused")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("finished normally", result.stdout)

    def test_large_results_cross_the_pipe(self):
        result = self._run_parent("child_sends_many_turns", "/unused")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("finished normally", result.stdout)

    def test_child_dying_without_a_result_twice_fails_the_run(self):
        result = self._run_parent("child_dies_silently", "/unused")
        self.assertEqual(1, result.returncode)
        self.assertEqual(1, result.stderr.count("retrying once in a fresh process"))
        self.assertEqual(
            "Error: diarization process exited with code 3 before returning a result",
            result.stderr.strip().splitlines()[-1],
        )
        self.assertNotIn("finished normally", result.stdout)

    def test_child_killed_once_is_retried(self):
        state = _state_file()
        result = self._run_parent("child_dies_once_then_succeeds", state)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("finished normally", result.stdout)
        self.assertIn(
            "Warning: diarization failed (process exited with code -9); "
            "retrying once in a fresh process",
            result.stderr,
        )
        self.assertEqual("2", _read(state))

    def test_out_of_memory_is_retried_once(self):
        state = _state_file()
        result = self._run_parent("child_out_of_memory_once", state)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("finished normally", result.stdout)
        self.assertIn(
            "Warning: diarization failed (OutOfMemoryError: CUDA out of memory.)",
            result.stderr,
        )
        self.assertEqual("2", _read(state))

    def test_second_failure_ends_the_run_with_its_own_error(self):
        state = _state_file()
        result = self._run_parent("child_out_of_memory_always", state)
        self.assertEqual(1, result.returncode)
        self.assertEqual(
            "OutOfMemoryError: CUDA out of memory (attempt 2).",
            result.stderr.strip().splitlines()[-1],
        )
        self.assertEqual("2", _read(state))
        self.assertNotIn("finished normally", result.stdout)

    def test_failure_sequential_would_share_is_not_retried(self):
        state = _state_file()
        result = self._run_parent("child_fails_for_good", state)
        self.assertEqual(1, result.returncode)
        self.assertEqual(
            "ValueError: broken pipeline config",
            result.stderr.strip().splitlines()[-1],
        )
        self.assertNotIn("retrying", result.stderr)
        self.assertEqual("1", _read(state))

    def test_release_waits_for_the_child_to_exit(self):
        result = self._run_parent("child_sends_turns", "/unused", "release")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("released; child alive: False", result.stdout)
        self.assertIn("finished normally", result.stdout)

    @unittest.skipUnless(
        HAS_PYANNOTE_AUDIO, "the real child imports torch and pyannote.audio"
    )
    def test_missing_config_keeps_the_error_text_the_platform_parses(self):
        missing = "/nonexistent/pyannote/audio/models/pyannote_diarization_config.yaml"
        result = self._run_parent("real", missing)
        self.assertEqual(1, result.returncode)
        last_line = result.stderr.strip().splitlines()[-1]
        self.assertEqual(
            f"FileNotFoundError: Config file not found: {missing}", last_line
        )
        self.assertNotIn("retrying", result.stderr)

    def test_child_dies_with_its_parent(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [REPO_ROOT] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
        )
        pid_file = tempfile.mktemp()
        parent = subprocess.Popen(
            [sys.executable, FIXTURES, "child_sleeps_forever", pid_file], env=env
        )
        try:
            deadline = time.time() + 60
            while not (os.path.exists(pid_file) and os.path.getsize(pid_file)):
                self.assertLess(time.time(), deadline, "child never started")
                time.sleep(0.1)
            with open(pid_file) as handle:
                child_pid = int(handle.read())
            self.assertTrue(_alive(child_pid))

            parent.send_signal(signal.SIGKILL)  # what a Process timeout does
            parent.wait()

            deadline = time.time() + 5
            while _alive(child_pid) and time.time() < deadline:
                time.sleep(0.05)
            self.assertFalse(_alive(child_pid), "diarization child outlived the CLI")
        finally:
            if parent.poll() is None:
                parent.kill()


@unittest.skipIf(sys.platform == "win32", "POSIX process semantics")
class TestTranscriptionOutOfMemory(unittest.TestCase):
    """main() with a fake Whisper: an OOM while pyannote holds memory is retried."""

    def _run_cli(self, *whisper_failures):
        output_dir = tempfile.mkdtemp()
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [REPO_ROOT] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
        )
        result = subprocess.run(
            [sys.executable, FIXTURES, "cli", output_dir, *whisper_failures],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            cwd=tempfile.gettempdir(),
        )
        srt = os.path.join(output_dir, "episode.srt")
        return result, (_read(srt) if os.path.exists(srt) else None)

    def test_out_of_memory_waits_for_diarization_then_retries(self):
        result, srt = self._run_cli("CUDA failed with error out of memory")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn(
            "Warning: transcription ran out of memory "
            "(CUDA failed with error out of memory); "
            "retrying once diarization has released its memory",
            result.stderr,
        )
        released = result.stdout.index("released; child alive: False")
        self.assertGreater(result.stdout.index("fake transcription ran"), released)
        self.assertIn("[SPEAKER_00]", srt)
        self.assertIn("hello", srt)

    def test_other_transcription_errors_are_not_retried(self):
        result, srt = self._run_cli("CUDA failed with error an illegal memory access")
        self.assertEqual(
            0, result.returncode, result.stderr
        )  # unchanged: per-file error
        self.assertIn("Error: unable to process file", result.stderr)
        self.assertNotIn("retrying", result.stderr)
        self.assertIsNone(srt)

    def test_a_second_out_of_memory_is_a_per_file_error(self):
        result, srt = self._run_cli(
            "CUDA failed with error out of memory",
            "CUDA failed with error out of memory",
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(1, result.stderr.count("retrying once diarization"))
        self.assertIn("Error: unable to process file", result.stderr)
        self.assertIsNone(srt)


def _state_file():
    handle, path = tempfile.mkstemp()
    os.close(handle)
    os.unlink(path)  # the fixtures count attempts from "absent"
    return path


def _read(path):
    with open(path) as handle:
        return handle.read()


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill(0); it holds no GPU memory, count it as gone.
    try:
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return True
    return bool(state) and not state.startswith("Z")


if __name__ == "__main__":
    unittest.main()
