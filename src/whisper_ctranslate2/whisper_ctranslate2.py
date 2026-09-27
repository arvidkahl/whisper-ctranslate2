import os
from .transcribe import Transcribe, TranscriptionOptions
from .languages import from_language_to_iso_code
import numpy as np
import warnings
from typing import Union, List
from .writers import get_writer
from .live import Live
import sys
import datetime
from .commandline import CommandLine
from .parallel_diarization import (
    PARALLEL,
    ParallelDiarization,
    diarization_mode,
    is_out_of_memory,
    mirror_sequential_parent_state,
)
from .speaker_assignment import assign_speakers_from_turns


def get_diarization(audio, diarize_model, verbose):
    diarization_output = {}
    for audio_path in audio:
        if verbose and len(audio) > 1:
            print(f"\nFile: '{audio_path}' (diarization)")

        start_time = datetime.datetime.now()
        diarize_segments = diarize_model.run_model(audio_path)
        diarization_output[audio_path] = diarize_segments
        if verbose:
            print(f"Time used for diarization: {datetime.datetime.now() - start_time}")

    diarize_model.unload_model()
    return diarization_output


def transcribe_file(
    transcribe, audio_path, task, language, verbose, options, parallel_diarization
):
    """Transcribe one file. If Whisper runs out of GPU memory while the parallel
    diarization process still holds its own, wait for that process to exit and
    try once more: with pyannote gone, Whisper has at least the memory the
    sequential path gives it."""
    try:
        return transcribe.inference(audio_path, task, language, verbose, False, options)
    except Exception as error:
        if parallel_diarization is None or not is_out_of_memory(error):
            raise
        sys.stderr.write(
            f"Warning: transcription ran out of memory ({error}); "
            "retrying once diarization has released its memory\n"
        )
        parallel_diarization.release()
        return transcribe.inference(audio_path, task, language, verbose, False, options)


def get_transcription_options(args):
    temperature = args.pop("temperature")

    if (increment := args.pop("temperature_increment_on_fallback")) is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = [temperature]

    suppress_tokens: str = args.pop("suppress_tokens")

    if suppress_tokens is None or len(suppress_tokens) == 0:
        suppress_tokens = []
    else:
        suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

    return TranscriptionOptions(
        beam_size=args.pop("beam_size"),
        best_of=args.pop("best_of"),
        patience=args.pop("patience"),
        length_penalty=args.pop("length_penalty"),
        repetition_penalty=args.pop("repetition_penalty"),
        no_repeat_ngram_size=args.pop("no_repeat_ngram_size"),
        log_prob_threshold=args.pop("logprob_threshold"),
        no_speech_threshold=args.pop("no_speech_threshold"),
        compression_ratio_threshold=args.pop("compression_ratio_threshold"),
        condition_on_previous_text=args.pop("condition_on_previous_text"),
        temperature=temperature,
        prompt_reset_on_temperature=args.pop("prompt_reset_on_temperature"),
        initial_prompt=args.pop("initial_prompt"),
        suppress_blank=args.pop("suppress_blank"),
        suppress_tokens=suppress_tokens,
        word_timestamps=args.pop("word_timestamps"),
        prepend_punctuations=args.pop("prepend_punctuations"),
        append_punctuations=args.pop("append_punctuations"),
        print_colors=args.pop("print_colors"),
        hallucination_silence_threshold=args.pop("hallucination_silence_threshold"),
        vad_filter=args.pop("vad_filter"),
        vad_threshold=args.pop("vad_threshold"),
        vad_min_speech_duration_ms=args.pop("vad_min_speech_duration_ms"),
        vad_max_speech_duration_s=args.pop("vad_max_speech_duration_s"),
        vad_min_silence_duration_ms=args.pop("vad_min_silence_duration_ms"),
    )


def get_language(language, model_directory, model):
    language = from_language_to_iso_code(language)

    if (
        not model_directory
        and model.endswith(".en")
        and language not in {"en", "English"}
    ):
        if language is not None:
            warnings.warn(
                f"{model} is an English-only model but receipted '{language}'; using English instead."
            )
        language = "en"

    return language


def main():
    args = CommandLine().read_command_line()
    output_dir: str = args.pop("output_dir")
    output_format: str = args.pop("output_format")
    os.makedirs(output_dir, exist_ok=True)
    model: str = args.pop("model")
    threads: int = args.pop("threads")
    language: str = args.pop("language")
    task: str = args.pop("task")
    device: str = args.pop("device")
    compute_type: str = args.pop("compute_type")
    verbose: bool = args.pop("verbose")
    model_directory: str = args.pop("model_directory")
    cache_directory: str = args.pop("model_dir")
    device_index: Union[int, List[int]] = args.pop("device_index")
    live_transcribe: bool = args.pop("live_transcribe")
    audio: str = args.pop("audio")
    local_files_only: bool = args.pop("local_files_only")
    live_volume_threshold: float = args.pop("live_volume_threshold")
    live_input_device: int = args.pop("live_input_device")
    hf_token = args.pop("hf_token")
    speaker_name = args.pop("speaker_name")

    language = get_language(language, model_directory, model)
    options = get_transcription_options(args)

    if not live_transcribe and len(audio) == 0:
        sys.stderr.write("You need to specify one or more audio files\n")
        sys.stderr.write(
            "Use `whisper-ctranslate2 --help` to see the available options.\n"
        )
        return

    word_options = [
        "highlight_words",
        "max_line_count",
        "max_line_width",
        "max_words_per_line",
    ]
    if not options.word_timestamps:
        for option in word_options:
            if args[option]:
                sys.stderr.write(f"--{option} requires --word_timestamps True\n")
                return

    if options.hallucination_silence_threshold and not options.word_timestamps:
        sys.stderr.write(
            "--hallucination_silence_threshold requires --word_timestamps True"
        )
        return

    if args["max_line_count"] and not args["max_line_width"]:
        warnings.warn("--max_line_count has no effect without --max_line_width")

    if args["max_words_per_line"] and args["max_line_width"]:
        warnings.warn("--max_words_per_line has no effect with --max_line_width")

    writer_options = list(word_options)
    writer_options.append("pretty_json")
    writer_args = {arg: args.pop(arg) for arg in writer_options}

    if not verbose and options.print_colors:
        sys.stderr.write("You cannot disable verbose and enable print colors\n")
        return

    if live_transcribe and not Live.is_available():
        Live.force_not_available_exception()

    if verbose and not language:
        if live_transcribe:
            print(
                "Consider specifying the language using `--language`. It improves significantly prediction in live transcription."
            )
        else:
            print(
                "Detecting language using up to the first 30 seconds. Use `--language` to specify the language"
            )

    if options.print_colors and output_dir and not options.word_timestamps:
        print(
            "Print colors requires word-level time stamps. Generated files in output directory will have word-level timestamps"
        )

    output_dir = os.path.abspath(output_dir)
    if model_directory:
        model_filename = os.path.join(model_directory, "model.bin")
        if not os.path.exists(model_filename):
            sys.stderr.write(f"Model file '{model_filename}' does not exists\n")
            return
        model_dir = model_directory
    else:
        model_dir = model

    if live_transcribe:
        Live(
            model_dir,
            cache_directory,
            local_files_only,
            task,
            language,
            threads,
            device,
            device_index,
            compute_type,
            verbose,
            live_volume_threshold,
            live_input_device,
            options,
        ).inference()

        return

    diarization = len(hf_token) > 0
    diarization_device = "cpu" if device == "auto" else device

    parallel_diarization = None
    if diarization and diarization_mode() == PARALLEL:
        # Started before the Whisper model loads: pyannote runs in its own
        # process, with its own CUDA context, while this one transcribes.
        parallel_diarization = ParallelDiarization(
            audio, hf_token, diarization_device, threads, verbose
        )
        parallel_diarization.start()

    transcribe = Transcribe(
        model_dir,
        device,
        device_index,
        compute_type,
        threads,
        cache_directory,
        local_files_only,
    )

    if parallel_diarization is not None:
        mirror_sequential_parent_state(diarization_device, threads)
    elif diarization:
        # Import is done here then dependencies like torch are only imported if we really need diarization
        from .diarization import Diarization

        diarize_model = Diarization(use_auth_token=hf_token, device=diarization_device)
        if threads > 0:
            diarize_model.set_threads(threads)

    diarization_output = {}
    if diarization and parallel_diarization is None:
        diarization_output = get_diarization(audio, diarize_model, verbose)

    # Sequential path (kill switch): every file is diarized before any is transcribed,
    # as the fork always did. The parallel path collects its turns in the loop below.
    for audio_path in audio:
        try:
            if verbose and len(audio) > 1:
                print(f"\nFile: '{audio_path} ({task})'")

            start_time = datetime.datetime.now()
            result = transcribe_file(
                transcribe,
                audio_path,
                task,
                language,
                verbose,
                options,
                parallel_diarization,
            )

            if diarization:
                if verbose:
                    print(
                        f"Time used for transcription: {datetime.datetime.now() - start_time}"
                    )
                if parallel_diarization is not None:
                    result = assign_speakers_from_turns(
                        parallel_diarization.turns_for(audio_path),
                        result,
                        speaker_name,
                    )
                else:
                    result = diarize_model.assign_speakers_to_segments(
                        diarization_output[audio_path], result, speaker_name
                    )

            writer = get_writer(output_format, output_dir)
            writer(result, audio_path, writer_args)

        except Exception as e:
            sys.stderr.write(
                f"Error: unable to process file: {audio_path} with exception '{e}'\n"
            )
            continue

    if parallel_diarization is not None:
        parallel_diarization.finish()

    if verbose:
        print(f"Transcription results written to '{output_dir}' directory")


if __name__ == "__main__":
    main()
