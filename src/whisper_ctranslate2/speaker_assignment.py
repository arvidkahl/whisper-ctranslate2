"""Speaker assignment shared by the sequential and the parallel diarization paths.

Deliberately free of torch and pyannote imports: the parallel path merges in the
CLI process, which never loads pyannote.
"""

import numpy as np
from collections import OrderedDict


class TurnSegment:
    """The two attributes of a pyannote ``Segment`` the assignment reads.

    Not a tuple on purpose: numpy would try to unpack a tuple into the
    structured ``segment`` field instead of storing it as one object.
    """

    __slots__ = ("start", "end")

    def __init__(self, start, end):
        self.start = start
        self.end = end


def annotation_to_turns(annotation):
    """Plain, picklable ``(start, end, track, speaker)`` tuples in itertracks() order."""
    return [
        (segment.start, segment.end, track, speaker)
        for segment, track, speaker in annotation.itertracks(yield_label=True)
    ]


def assign_speakers_from_turns(turns, transcript_result, speaker_name):
    """Same assignment as the sequential path, fed from serialized turns."""
    diarize_data = [
        (TurnSegment(start, end), track, speaker)
        for start, end, track, speaker in turns
    ]
    return assign_speakers(diarize_data, transcript_result, speaker_name)


def assign_speakers(diarize_data, transcript_result, speaker_name):
    """Label each transcript segment with the speaker it overlaps most.

    Moved verbatim from ``Diarization._do_assign_speakers_to_segments``.
    """
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
