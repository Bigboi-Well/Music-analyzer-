"""
Pro Music Analyzer & Playlist Generator — rewritten, patched, runnable.

How to run:
1) python -m venv venv
2) venv\Scripts\activate  (Windows)  OR  source venv/bin/activate (mac/linux)
3) pip install streamlit librosa soundfile numpy pandas matplotlib scipy
4) streamlit run app.py
"""

import io
import hashlib
import traceback
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------
# Compatibility patches
# ----------------------------
# Fix for SciPy versions where scipy.signal.hann was removed/moved
import scipy.signal
try:
    # prefer the windows.hann if available
    if not hasattr(scipy.signal, "hann"):
        from scipy.signal import windows
        scipy.signal.hann = windows.hann
except Exception:
    # fallback to numpy hanning
    if not hasattr(scipy.signal, "hann"):
        scipy.signal.hann = np.hanning  # simple fallback

# Import librosa after patch
import librosa
import librosa.display

# Fix waveshow color bug (matplotlib/librosa compatibility)
_old_waveshow = librosa.display.waveshow
def _waveshow_patched(*args, **kwargs):
    kwargs.setdefault("color", "b")
    return _old_waveshow(*args, **kwargs)
librosa.display.waveshow = _waveshow_patched

# ----------------------------
# App config
# ----------------------------
st.set_page_config(page_title="🎵 Pro Music Analyzer", page_icon="🎵", layout="wide")
st.title("🎵 Pro Music Analyzer — Rewritten & Patched")
st.caption("Analyze audio (tempo/key/mood/spectral), visualize, compare and generate playlists.")

# ----------------------------
# Constants & templates
# ----------------------------
PITCH_CLASS_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
MAJOR_TEMPLATES = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
MINOR_TEMPLATES = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
MOOD_LABELS = ["Rap/Hip-Hop","Reggaeton/Dance","Pop","EDM/Dance","Chill/Ambient","Dark/Heavy","Neutral"]

SUPPORTED_EXT = ("wav","mp3","ogg","flac","m4a","mp4","aac")

# ----------------------------
# Helper functions
# ----------------------------
def file_hash(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def rotate(a: np.ndarray, n: int) -> np.ndarray:
    return np.roll(a, n)

def detect_key(chroma_mean: np.ndarray) -> str:
    # Compare to Krumhansl templates
    major_scores = [np.corrcoef(chroma_mean, rotate(MAJOR_TEMPLATES, i))[0,1] for i in range(12)]
    minor_scores = [np.corrcoef(chroma_mean, rotate(MINOR_TEMPLATES, i))[0,1] for i in range(12)]
    if np.max(major_scores) >= np.max(minor_scores):
        pc = int(np.argmax(major_scores))
        mode = "major"
    else:
        pc = int(np.argmax(minor_scores))
        mode = "minor"
    return f"{PITCH_CLASS_NAMES[pc]} {mode}"

def classify_mood_extended(tempo: float, rms: float, centroid: float, rolloff: float, zcr: float) -> str:
    """
    Rule-based multi-label mood classifier tuned to avoid marking rap/reggaeton as Neutral.
    Priority checks for Rap and Reggaeton patterns first.
    """
    # Rap / Hip-Hop: strong percussive energy, mid tempo, high ZCR or high RMS with moderate centroid
    if (80 <= tempo <= 110 and (rms > 0.05 or zcr > 0.06) and 2000 <= centroid <= 4000):
        return "Rap/Hip-Hop"
    # Reggaeton / Dance pop: percussive + rolloff high + tempo moderate
    if (90 <= tempo <= 120 and rolloff > 5000 and zcr > 0.06):
        return "Reggaeton/Dance"
    # EDM / Dance: fast tempo and bright
    if tempo > 130 and centroid > 3500:
        return "EDM/Dance"
    # Pop: mid tempo bright
    if 100 <= tempo <= 130 and centroid > 2500:
        return "Pop"
    # Chill/Ambient: low RMS, low centroid
    if rms < 0.03 and centroid < 2000:
        return "Chill/Ambient"
    # Dark/Heavy: loud but mellow centroid
    if rms > 0.06 and centroid < 2200:
        return "Dark/Heavy"
    # fallback neutral
    return "Neutral"

def safe_librosa_load(data: bytes, sr=22050, mono=True, duration=None):
    try:
        y, s = librosa.load(io.BytesIO(data), sr=sr, mono=mono, duration=duration)
        # ensure float32
        if not isinstance(y, np.ndarray):
            y = np.array(y, dtype=np.float32)
        else:
            y = y.astype(np.float32)
        return y, s
    except Exception as e:
        raise RuntimeError(f"librosa.load failed: {e}")

def compute_sections(y: np.ndarray, sr: int, n_peaks: int = 6) -> List[float]:
    # onset strength then peak pick — use keyword args for compatibility
    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    try:
        peaks = librosa.util.peak_pick(oenv,
                                      pre_max=16, post_max=16,
                                      pre_avg=16, post_avg=16,
                                      delta=0.6 * (oenv.max() if oenv.size>0 else 1.0),
                                      wait=5)
    except TypeError:
        # older librosa might accept positional args; if fails, call with default peak_pick behavior
        peaks = librosa.util.peak_pick(oenv)
    times = librosa.frames_to_time(peaks, sr=sr) if peaks.size>0 else np.array([])
    boundaries = [0.0] + list(times[:n_peaks]) + [len(y)/sr]
    return sorted(list(dict.fromkeys([round(float(x),2) for x in boundaries])))

# ----------------------------
# Analysis (cached)
# ----------------------------
@st.cache_data(show_spinner=False)
def analyze_bytes(data: bytes, sr_target: int = 22050, duration_preview: float = 30.0) -> Dict[str, Any]:
    """
    Analyze audio bytes and return a dict of computed metrics and figures.
    Uses a preview duration for speed (unless you pass full file).
    """
    # Load a preview (trim silence)
    y, sr = safe_librosa_load(data, sr=sr_target, mono=True, duration=duration_preview)
    if y.size == 0:
        raise RuntimeError("Empty audio after loading.")
    yt, _ = librosa.effects.trim(y, top_db=30)

    # Core features
    tempo, beat_frames = librosa.beat.beat_track(y=yt, sr=sr)
    tempo = float(tempo)

    rms = librosa.feature.rms(y=yt)[0]
    rms_mean = float(np.mean(rms))

    centroid = np.mean(librosa.feature.spectral_centroid(y=yt, sr=sr)[0])
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=yt, sr=sr)[0])
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=yt, sr=sr)[0])
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=yt)[0]))

    # chroma -> key
    chroma = librosa.feature.chroma_cqt(y=yt, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    key_name = detect_key(chroma_mean)

    # HPSS heuristic (harmonic/percussive)
    try:
        D = librosa.stft(yt, n_fft=2048)
        H, P = librosa.decompose.hpss(D)
        harmonic_energy = float(np.sum(np.abs(H)))
        percussive_energy = float(np.sum(np.abs(P)))
    except Exception:
        # fallback to time-domain HPSS
        y_h, y_p = librosa.effects.hpss(yt)
        harmonic_energy = float(np.sum(np.abs(y_h)))
        percussive_energy = float(np.sum(np.abs(y_p)))

    hprs_ratio = harmonic_energy / (harmonic_energy + percussive_energy + 1e-12)

    # Mood
    mood = classify_mood_extended(tempo, rms_mean, centroid, rolloff, zcr)

    # Figures (render and return handles)
    figs = {}
    try:
        figs["waveform"] = render_waveform_fig(yt, sr)
        figs["melspec"] = render_melspectrogram_fig(yt, sr)
        figs["chromagram"] = render_chromagram_fig(chroma, sr)
    except Exception as e:
        # If figure generation fails, store the exception message
        figs["error"] = str(e)

    duration = librosa.get_duration(y=y, sr=sr)

    return {
        "duration": float(duration),
        "tempo": tempo,
        "rms_mean": rms_mean,
        "centroid": float(centroid),
        "bandwidth": float(bandwidth),
        "rolloff": float(rolloff),
        "zcr": float(zcr),
        "key": key_name,
        "harmonic_energy": harmonic_energy,
        "percussive_energy": percussive_energy,
        "hprs_ratio": float(hprs_ratio),
        "mood": mood,
        "figs": figs,
        "sr": sr
    }

# ----------------------------
# Figure renderers (helpers)
# ----------------------------
def render_waveform_fig(y: np.ndarray, sr: int):
    fig, ax = plt.subplots(figsize=(8,2.2))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    return fig

def render_melspectrogram_fig(y: np.ndarray, sr: int):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(8,2.6))
    img = librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel', ax=ax)
    ax.set_title("Mel Spectrogram (dB)")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig

def render_chromagram_fig(chroma, sr):
    fig, ax = plt.subplots(figsize=(8,2.2))
    img = librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', ax=ax)
    ax.set_title("Chromagram")
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    return fig

# ----------------------------
# UI: Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("Playlist Criteria")
    target_mood = st.selectbox("Target mood", MOOD_LABELS, index=2)
    target_bpm = st.slider("Target tempo (BPM)", min_value=60, max_value=200, value=120)
    target_key_pc = st.selectbox("Target key (pitch class)", PITCH_CLASS_NAMES, index=0)
    target_mode = st.selectbox("Mode", ["major", "minor"], index=0)
    playlist_size = st.slider("Playlist size", min_value=1, max_value=20, value=6)
    st.write("---")
    st.caption("This tool was developed as a workshop group project under professor supervision.")

# ----------------------------
# UI: upload files
# ----------------------------
uploaded = st.file_uploader("Upload audio files (wav/mp3/ogg/flac/m4a)", accept_multiple_files=True)

if not uploaded:
    st.info("Upload one or more audio files to analyze.")
    st.stop()

# ----------------------------
# Analyze uploaded files
# ----------------------------
rows = []
fig_store: Dict[str, Dict[str, Any]] = {}

for f in uploaded:
    name = f.name
    ext_ok = any(name.lower().endswith("." + e) for e in SUPPORTED_EXT)
    if not ext_ok:
        st.warning(f"Skipping unsupported file type: {name}")
        continue
    data = f.read()
    key = file_hash(data)
    try:
        res = analyze_bytes(data)
    except Exception as e:
        st.error(f"Failed to analyze {name}: {e}\n{traceback.format_exc()}")
        continue
    fig_store[name] = res["figs"]
    # parse key and mode
    try:
        pc, mode = res["key"].split()
    except Exception:
        pc, mode = res["key"], "major"
    rows.append({
        "filename": name,
        "duration_s": round(res["duration"], 2),
        "tempo_bpm": round(res["tempo"], 1),
        "rms": round(res["rms_mean"], 5),
        "centroid": round(res["centroid"], 2),
        "bandwidth": round(res["bandwidth"], 2),
        "rolloff": round(res["rolloff"], 2),
        "zcr": round(res["zcr"], 5),
        "key_pc": pc,
        "mode": mode,
        "full_key": res["key"],
        "hprs": round(res["hprs_ratio"], 3),
        "mood": res["mood"],
    })

if len(rows) == 0:
    st.error("No valid audio files found after upload.")
    st.stop()

df = pd.DataFrame(rows)

# ----------------------------
# Show summary table and export
# ----------------------------
st.subheader("Analysis Summary")
st.dataframe(df, use_container_width=True)
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv_bytes, file_name="audio_analysis.csv", mime="text/csv")

# ----------------------------
# Visualizations per track (tabs)
# ----------------------------
st.subheader("Track Visualizations")
tabs = st.tabs(list(fig_store.keys()))
for tab, (name, figs) in zip(tabs, fig_store.items()):
    with tab:
        col1, col2 = st.columns([1,1])
        with col1:
            if "waveform" in figs:
                st.pyplot(figs["waveform"])
            if "chromagram" in figs:
                st.pyplot(figs["chromagram"])
        with col2:
            if "melspec" in figs:
                st.pyplot(figs["melspec"])
        # audio playback
        st.audio(uploaded[[i for i,u in enumerate(uploaded) if u.name==name][0]].getvalue())

# ----------------------------
# Playlist generation
# ----------------------------
st.subheader("Generate Playlist")
# Build target vector (bpm, rms approx, centroid)
def target_vector(mood_label):
    if mood_label == "Energetic":
        return np.array([target_bpm, 0.065, 3200.0])
    if mood_label == "EDM/Dance":
        return np.array([target_bpm, 0.07, 3600.0])
    if mood_label == "Rap/Hip-Hop":
        return np.array([target_bpm, 0.055, 3200.0])
    if mood_label == "Reggaeton/Dance":
        return np.array([target_bpm, 0.055, 3000.0])
    if mood_label == "Pop":
        return np.array([target_bpm, 0.052, 3000.0])
    if mood_label == "Chill/Ambient":
        return np.array([target_bpm, 0.035, 2200.0])
    return np.array([target_bpm, 0.05, 2800.0])

tv = target_vector(target_mood)

# filter by key if possible
mask_key = (df["key_pc"] == target_key_pc) & (df["mode"] == target_mode)
df_key = df[mask_key]
if df_key.empty:
    df_key = df.copy()

# distance metric: normalized Euclidean + mood penalty
MOOD_ORDER = {m: i for i, m in enumerate(MOOD_LABELS)}
def row_distance(r):
    v = np.array([r["tempo_bpm"], r["rms"], r["centroid"]])
    scales = np.array([1.0, 200.0, 0.01])  # scale features
    dist = np.linalg.norm((v - tv) * scales)
    mood_pen = abs(MOOD_ORDER.get(r["mood"], 3) - MOOD_ORDER.get(target_mood, 3))
    return float(dist + mood_pen)

df_key = df_key.copy()
df_key["distance"] = df_key.apply(row_distance, axis=1)
playlist_df = df_key.sort_values(["distance", "tempo_bpm"]).head(playlist_size)

st.markdown(f"**Target:** {target_mood} · {target_bpm} BPM · {target_key_pc} {target_mode}")
st.dataframe(playlist_df[["filename","mood","tempo_bpm","full_key","distance"]].reset_index(drop=True), use_container_width=True)

# m3u (filenames only)
m3u = "#EXTM3U\n" + "\n".join(playlist_df["filename"].tolist())
st.download_button("Download M3U (filenames)", data=m3u.encode("utf-8"), file_name="playlist.m3u", mime="audio/x-mpegurl")

st.success("Playlist generated. Tweak playlist criteria on the sidebar to refine.")
