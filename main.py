import json
import base64
import math
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import io
import soundfile as sf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def freq_to_midi(freq):
    if freq <= 0:
        return 0
    return 69.0 + 12.0 * np.log2(freq / 440.0)

def midi_to_note_name(midi):
    """Convert MIDI number to human-readable note name like C4, D#5."""
    if midi is None or midi <= 0:
        return "—"
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note_idx = int(round(midi)) % 12
    octave = int(round(midi)) // 12 - 1
    return f"{note_names[note_idx]}{octave}"

PITCH_TOLERANCE_SEMITONES = 0.5   # 50 cents — standard AP grading tolerance
HOP_LENGTH = 512

def preprocess_for_pitch(y):
    """Light preprocessing to reduce rumble/static before pitch tracking."""
    y = librosa.effects.preemphasis(y, coef=0.97)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y

def average_note_pitch(y, sr, window_start, window_end, hop_length=HOP_LENGTH):
    """
    Collapse a note window to one pitch by averaging voiced frames.
    Rejects noisy/static frames via RMS and spectral flatness gating.
    Returns (detected_midi, detected_onset) or (None, None).
    """
    win_start = max(0.0, window_start)
    win_end = min(len(y) / sr, window_end)
    if win_end - win_start < 0.05:
        return None, None

    i0 = int(win_start * sr)
    i1 = int(win_end * sr)
    if i1 - i0 < int(sr * 0.04):
        return None, None

    y_clip = preprocess_for_pitch(y[i0:i1])

    f0 = librosa.yin(
        y_clip,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C6'),
        sr=sr,
        hop_length=hop_length,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length) + win_start

    rms = librosa.feature.rms(y=y_clip, hop_length=hop_length)[0]
    if len(rms) == 0:
        return None, None
    rms_norm = rms / (np.max(rms) + 1e-9)

    flatness = librosa.feature.spectral_flatness(y=y_clip, hop_length=hop_length)[0]
    flatness_norm = flatness / (np.max(flatness) + 1e-9)

    rms_times = librosa.frames_to_time(np.arange(len(rms_norm)), sr=sr, hop_length=hop_length)

    voiced_freqs = []
    voiced_times = []
    for i, t in enumerate(times):
        if np.isnan(f0[i]) or f0[i] <= 0:
            continue

        local_t = t - win_start
        rms_val = float(np.interp(local_t, rms_times, rms_norm))
        flat_idx = min(i, len(flatness_norm) - 1)
        flat_val = float(flatness_norm[flat_idx])

        # Skip quiet frames and highly noisy/buzzy frames
        if rms_val < 0.1:
            continue
        if flat_val > 0.8:
            continue

        voiced_freqs.append(float(f0[i]))
        voiced_times.append(float(t))

    if len(voiced_freqs) < 3:
        return None, None

    midis = np.array([freq_to_midi(f) for f in voiced_freqs])
    lo = np.percentile(midis, 20)
    hi = np.percentile(midis, 80)
    trimmed = midis[(midis >= lo) & (midis <= hi)]
    if len(trimmed) < 2:
        trimmed = midis

    detected_midi = float(np.mean(trimmed))

    return detected_midi, voiced_times


def build_note_matches(expected_notes, sung_segments, beat_sec):
    """
    Match expected notes to sung segments in score order using timing only.
    Pitch is graded separately so rhythm credit doesn't require pitch accuracy.
    Returns (matches, global_rhythm_offset).
    """
    matches = []
    offset_samples = []
    sung_used = set()
    last_sung_start = -1.0

    for exp in expected_notes:
        exp_start = exp["start"]
        exp_end = exp_start + exp["duration"]
        search_lo = exp_start - beat_sec * 1.0
        search_hi = exp_end + beat_sec * 0.75

        best_idx = None
        best_err = float("inf")

        for i, sn in enumerate(sung_segments):
            if i in sung_used:
                continue
            if sn["start"] < search_lo or sn["start"] > search_hi:
                continue
            if sn["start"] < last_sung_start - 0.05:
                continue

            time_err = abs(sn["start"] - exp_start)
            if time_err < best_err:
                best_err = time_err
                best_idx = i

        if best_idx is not None:
            sn = sung_segments[best_idx]
            sung_used.add(best_idx)
            last_sung_start = sn["start"]
            offset_samples.append(sn["start"] - exp_start)
            matches.append({"exp": exp, "sung": sn})

    global_offset = float(np.median(offset_samples)) if offset_samples else 0.0
    return matches, global_offset

HM_OVERHANG_SEC = 0.5
RHYTHM_LENIENCY_BEATS = 0.5   # AP graders expect notes within half a beat

def rhythm_tolerance_for_note(exp, beat_sec):
    return max(beat_sec * RHYTHM_LENIENCY_BEATS, exp["duration"] * 0.4, 0.08)

def grade_note_rhythm(exp, sung_segments, global_offset, beat_sec, matched_sung=None):
    """
    Decide if a note's rhythm is correct using the matched onset.
    Falls back to a nearby unmatched segment if no match was found.
    """
    exp_start = exp["start"]
    tolerance = rhythm_tolerance_for_note(exp, beat_sec)

    candidates = []
    if matched_sung is not None:
        candidates.append(matched_sung["start"])

    # Secondary: any segment near the expected onset (timing-based only)
    if not candidates:
        search_lo = exp_start - beat_sec * 0.75
        search_hi = exp_start + exp["duration"] + beat_sec * 0.5
        for sn in sung_segments:
            if search_lo <= sn["start"] <= search_hi:
                candidates.append(sn["start"])

    if not candidates:
        return False, None

    best_onset = None
    best_error = float("inf")
    for onset in candidates:
        aligned_onset = onset - global_offset
        error = abs(aligned_onset - exp_start)
        if error < best_error:
            best_error = error
            best_onset = onset

    return best_error <= tolerance, best_onset

def segment_sung_notes(y, sr, hop_length=512):
    """Extract voiced pitch segments from audio."""
    f0 = librosa.yin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C6'),
        sr=sr,
        hop_length=hop_length,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    if np.max(rms) > 0:
        rms = rms / np.max(rms)
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    sung_notes = []
    is_voiced = False
    current_note_start = 0.0
    current_note_f0 = []

    for i, t in enumerate(times):
        rms_val = float(np.interp(t, rms_times, rms))
        if not np.isnan(f0[i]) and rms_val > 0.05:
            if not is_voiced:
                is_voiced = True
                current_note_start = t
                current_note_f0 = [f0[i]]
            else:
                current_note_f0.append(f0[i])
        elif is_voiced:
            is_voiced = False
            if len(current_note_f0) > 3:
                median_freq = np.median(current_note_f0)
                duration = t - current_note_start
                if duration > 0.1:
                    sung_notes.append({
                        "start": float(current_note_start),
                        "duration": float(duration),
                        "midi": float(freq_to_midi(median_freq)),
                    })

    if is_voiced and len(current_note_f0) > 3:
        duration = float(times[-1]) - current_note_start
        if duration > 0.1:
            median_freq = np.median(current_note_f0)
            sung_notes.append({
                "start": float(current_note_start),
                "duration": float(duration),
                "midi": float(freq_to_midi(median_freq)),
            })

    return sung_notes

def half_measure_claim_window(hm_start, hm_end):
    """Half-measure territory plus 0.5s overhang on each side for grading."""
    return hm_start - HM_OVERHANG_SEC, hm_end + HM_OVERHANG_SEC

def notes_for_half_measure(notes_data, hm_start, hm_end, graded_starts):
    """Notes in this half-measure's claim window that have not been graded yet."""
    claim_start, claim_end = half_measure_claim_window(hm_start, hm_end)
    return [
        n for n in notes_data
        if claim_start <= n["start"] < claim_end and n["start"] not in graded_starts
    ]

def median_midi_in_window(y, sr, start, end, hop_length=HOP_LENGTH):
    detected, _ = average_note_pitch(y, sr, start, end, hop_length=hop_length)
    return detected

def compute_pitch_shift(notes_data, sung_notes, y, sr):
    """
    Find the best global transposition so the sung melody matches the written one.
    Uses median offset across all matched note pairs — more robust than first-note only.
    AP grading is relative pitch: the singer may start on any pitch.
    """
    if not notes_data or not sung_notes:
        return 0

    offsets = []
    for exp in notes_data:
        best_sn = None
        best_err = float("inf")
        for sn in sung_notes:
            err = abs(sn["start"] - exp["start"])
            if err < best_err and err < 1.5:
                best_err = err
                best_sn = sn
        if best_sn is not None:
            offsets.append(best_sn["midi"] - exp["midi"])

    if not offsets:
        return 0

    return int(round(float(np.median(offsets))))

def encode_wav_base64(y, sr):
  buf = io.BytesIO()
  sf.write(buf, y, sr, format='WAV')
  return base64.b64encode(buf.getvalue()).decode('ascii')

def note_analysis_window(exp, analysis_start, analysis_end):
    """Clip expected note times to the half-measure analysis window."""
    exp_start = exp["start"]
    exp_end = exp["start"] + exp["duration"]

    win_start = max(exp_start, analysis_start)
    win_end = min(exp_end, analysis_end)
    if win_end <= win_start:
        win_start = exp_start
        win_end = exp_end
    return win_start, win_end

def grade_half_measure(hm, notes_data, y, sr, beat_sec, sung_segments, graded_starts):
    """Grade one half-measure independently with audio overhang and deduplicated notes."""
    hm_start = hm["start"]
    hm_end = hm_start + hm["duration"]
    audio_start, audio_end = half_measure_claim_window(hm_start, hm_end)

    exp_notes = notes_for_half_measure(notes_data, hm_start, hm_end, graded_starts)
    if not exp_notes:
        return {
            "index": hm["index"],
            "label": hm.get("label", f"M{hm.get('measure', 1)}{'a' if hm.get('half', 1) == 1 else 'b'}"),
            "measure": hm.get("measure", 1),
            "half": hm.get("half", 1),
            "start": hm_start,
            "duration": hm["duration"],
            "expected_notes": [],
            "detected_summary": "—",
            "pitch_correct": False,
            "rhythm_correct": False,
            "correct": False,
            "notes_detail": [],
            "graded_starts": graded_starts,
        }

    local_matches, rhythm_offset = build_note_matches(exp_notes, sung_segments, beat_sec)
    match_by_exp_start = {m["exp"]["start"]: m["sung"] for m in local_matches}

    note_results = []
    for exp in exp_notes:
        exp_start = exp["start"]
        win_start, win_end = note_analysis_window(exp, audio_start, audio_end)

        pitch_correct = False
        rhythm_correct = False
        detected_midi = None

        detected_midi, _ = average_note_pitch(y, sr, win_start, win_end)
        if detected_midi is not None:
            if abs(detected_midi - exp["midi"]) <= PITCH_TOLERANCE_SEMITONES:
                pitch_correct = True

        matched_sung = match_by_exp_start.get(exp_start)
        rhythm_correct, _ = grade_note_rhythm(
            exp,
            sung_segments,
            rhythm_offset,
            beat_sec,
            matched_sung={"start": matched_sung["start"]} if matched_sung else None,
        )

        note_results.append({
            "expected_note": exp["note"],
            "expected_midi": exp["midi"],
            "expected_note_name": midi_to_note_name(exp["midi"]),
            "detected_midi": float(detected_midi) if detected_midi is not None else None,
            "detected_note_name": midi_to_note_name(detected_midi) if detected_midi is not None else "—",
            "pitch_correct": pitch_correct,
            "rhythm_correct": rhythm_correct,
            "start": exp_start,
            "duration": exp["duration"],
        })

    pitch_ok = sum(1 for r in note_results if r["pitch_correct"])
    rhythm_ok = sum(1 for r in note_results if r["rhythm_correct"])
    total = len(note_results)
    majority = total / 2.0

    pitch_correct_hm = pitch_ok >= majority
    rhythm_correct_hm = rhythm_ok >= majority
    correct_hm = pitch_correct_hm and rhythm_correct_hm

    detected_names = [r["detected_note_name"] for r in note_results if r["detected_midi"] is not None]
    detected_summary = " ".join(detected_names) if detected_names else "—"
    expected_names = [r["expected_note_name"] for r in note_results]
    new_graded_starts = graded_starts | {n["start"] for n in exp_notes}

    return {
        "index": hm["index"],
        "label": hm.get("label", f"M{hm.get('measure', 1)}{'a' if hm.get('half', 1) == 1 else 'b'}"),
        "measure": hm.get("measure", 1),
        "half": hm.get("half", 1),
        "start": hm_start,
        "duration": hm["duration"],
        "expected_notes": expected_names,
        "expected_summary": " ".join(expected_names),
        "detected_summary": detected_summary,
        "pitch_correct": pitch_correct_hm,
        "rhythm_correct": rhythm_correct_hm,
        "correct": correct_hm,
        "notes_detail": note_results,
        "pitch_score": f"{pitch_ok}/{total}",
        "rhythm_score": f"{rhythm_ok}/{total}",
        "graded_starts": new_graded_starts,
    }

@app.post("/api/grade")
async def grade_audio(
    audio: UploadFile,
    expected_notes: str = Form(...),
    half_measures: str = Form("[]"),
    meter: str = Form("4/4"),
    bpm: int = Form(80),
):
    try:
        notes_data = json.loads(expected_notes)
        hm_data = json.loads(half_measures)

        content = await audio.read()
        y, sr = sf.read(io.BytesIO(content))
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)

        sung_notes_raw = segment_sung_notes(y, sr)
        shift = compute_pitch_shift(notes_data, sung_notes_raw, y, sr)

        if shift != 0:
            y_aligned = librosa.effects.pitch_shift(y, sr=sr, n_steps=-shift)
        else:
            y_aligned = y

        sung_notes = segment_sung_notes(y_aligned, sr)

        beat_sec = 60.0 / bpm
        notes_aligned = [{**n, "midi": n["midi"]} for n in notes_data]

        if not hm_data:
            beats_per_measure = int(meter.split('/')[0])
            if meter == '6/8':
                beats_per_measure = 2
            half_measure_sec = (beats_per_measure / 2.0) * beat_sec
            total_dur = max((n["start"] + n["duration"] for n in notes_data), default=0)
            num_hm = max(1, int(math.ceil(total_dur / half_measure_sec)))
            hm_data = []
            for i in range(num_hm):
                measure_num = i // 2 + 1
                half_num = (i % 2) + 1
                hm_data.append({
                    "index": i,
                    "label": f"M{measure_num}{'a' if half_num == 1 else 'b'}",
                    "measure": measure_num,
                    "half": half_num,
                    "start": i * half_measure_sec,
                    "duration": half_measure_sec,
                })

        half_measure_results = []
        graded_starts = set()
        for hm in hm_data:
            result = grade_half_measure(
                hm,
                notes_aligned,
                y_aligned,
                sr,
                beat_sec,
                sung_notes,
                graded_starts,
            )
            graded_starts = result.pop("graded_starts", graded_starts)
            if result["notes_detail"] or hm_data:
                half_measure_results.append(result)

        hm_score = sum(1 for r in half_measure_results if r["correct"])
        num_half_measures = len(half_measure_results)

        pitch_correct_count = sum(1 for r in half_measure_results if r["pitch_correct"])
        flow_score = 1 if pitch_correct_count / max(num_half_measures, 1) >= 0.5 else 0
        total_score = hm_score + flow_score

        score_breakdown = {
            "Half-Measures": f"{hm_score} / {num_half_measures}",
            "Flow": f"{flow_score} / 1",
        }

        aligned_audio_b64 = encode_wav_base64(y_aligned, sr)

        return {
            "half_measure_results": half_measure_results,
            "results": half_measure_results,
            "sung_notes": sung_notes,
            "pitch_shift_semitones": shift,
            "aligned_audio_base64": aligned_audio_b64,
            "ap_score": total_score,
            "max_score": num_half_measures + 1,
            "score_breakdown": score_breakdown,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
