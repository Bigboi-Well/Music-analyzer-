import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import io
import scipy.signal

# Patch missing hann window in SciPy
if not hasattr(scipy.signal, "hann"):
    from scipy.signal import windows
    scipy.signal.hann = windows.hann

# Patch librosa waveshow color bug
_old_waveshow = librosa.display.waveshow
def _waveshow_patched(*args, **kwargs):
    kwargs.setdefault("color", "b")
    return _old_waveshow(*args, **kwargs)
librosa.display.waveshow = _waveshow_patched

# Mood classification logic
def classify_mood(tempo, energy, spectral_centroid, zero_crossings):
    if tempo < 80 and energy < 0.04:
        return "🧘 Calm / Chill"
    elif 80 <= tempo < 110 and 0.04 <= energy < 0.06:
        return "😌 Neutral / Mellow"
    elif 110 <= tempo < 140 and energy >= 0.05:
        return "💃 Dance / Happy"
    elif tempo >= 140 or spectral_centroid > 3500 or zero_crossings > 0.15:
        return "🔥 Energetic / Intense (Rap / Rock / EDM)"
    else:
        return "🎵 Neutral"

# Audio analyzer
def analyze_audio(file_bytes, filename):
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=None, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    energy = float(np.mean(rms))
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    zero_crossings = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    mood = classify_mood(tempo, energy, spectral_centroid, zero_crossings)

    return {
        "filename": filename,
        "y": y,
        "sr": sr,
        "tempo": tempo,
        "energy": energy,
        "spectral_centroid": spectral_centroid,
        "zero_crossings": zero_crossings,
        "mood": mood,
    }

# Streamlit UI
st.set_page_config(page_title="🎵 Pro Music Analyzer", layout="wide")

st.title("🎶 Pro Music Analyzer")
st.markdown("Upload a song and let the analyzer reveal its hidden musical personality. Built with ❤️ by students under professor guidance.")

uploaded_file = st.file_uploader("🎼 Upload a song (MP3/WAV)", type=["mp3", "wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    st.success(f"Analyzing **{uploaded_file.name}** ... Please wait ⏳")

    result = analyze_audio(uploaded_file.read(), uploaded_file.name)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Song Features")
        st.metric("Tempo (BPM)", f"{result['tempo']:.2f}")
        st.metric("Energy (RMS)", f"{result['energy']:.4f}")
        st.metric("Spectral Centroid", f"{result['spectral_centroid']:.2f} Hz")
        st.metric("Zero Crossing Rate", f"{result['zero_crossings']:.4f}")
        st.markdown(f"**🎭 Mood Spectrum:** {result['mood']}")

    with col2:
        st.subheader("📈 Waveform")
        fig, ax = plt.subplots(figsize=(8, 3))
        librosa.display.waveshow(result["y"], sr=result["sr"], ax=ax)
        ax.set_title("Waveform")
        st.pyplot(fig)

    st.subheader("🔊 Spectral Centroid Over Time")
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    spec_cent = librosa.feature.spectral_centroid(y=result["y"], sr=result["sr"])
    ax2.semilogy(spec_cent.T, label="Spectral Centroid", color="r")
    ax2.set_ylabel("Hz")
    ax2.set_title("Spectral Centroid")
    ax2.legend()
    st.pyplot(fig2)

else:
    st.info("👆 Upload a song to begin your musical journey.")
