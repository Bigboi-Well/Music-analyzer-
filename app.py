import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io

# App configuration
st.set_page_config(page_title="🎵 Pro Music Analyzer", page_icon="🎵")
st.title("🎵 Pro Music Analyzer")
st.write("Upload audio files and get detailed analysis: tempo, key, spectral features, mood, energy, and visualizations!")

# File uploader
uploaded_files = st.file_uploader(
    "Upload audio files",
    type=["mp3", "wav", "ogg"],
    accept_multiple_files=True
)

# Option: full song vs first 30 seconds
analyze_full_song = st.checkbox("Analyze full song (slower)", value=False)

# Cache analysis to speed up repeated runs
@st.cache_data
def analyze_audio(file_bytes, filename, full_song):
    y, sr = librosa.load(
        io.BytesIO(file_bytes),
        sr=22050,  # downsample to speed up
        mono=True,
        duration=None if full_song else 30
    )

    # Tempo and beats
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo = round(float(tempo), 2)

    # Key detection
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    key_idx = chroma.mean(axis=1).argmax()
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key = keys[key_idx]

    # Basic RMS energy
    rms = librosa.feature.rms(y=y).mean()

    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean()
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()

    # Mood detection (improved)
    if rms > 0.05 and spectral_centroid > 3000:
        mood = "Energetic"
    elif rms < 0.03 and spectral_centroid < 2000:
        mood = "Calm"
    else:
        mood = "Neutral"

    return {
        "name": filename,
        "tempo": tempo,
        "key": key,
        "mood": mood,
        "rms": round(float(rms), 4),
        "spectral_centroid": round(float(spectral_centroid), 2),
        "spectral_bandwidth": round(float(spectral_bandwidth), 2),
        "spectral_contrast": round(float(spectral_contrast), 2),
        "zero_crossing_rate": round(float(zero_crossing_rate), 4),
        "y": y,
        "sr": sr,
        "beats": beats
    }

audio_features = []

# Process uploaded files
if uploaded_files:
    st.subheader("Audio Analysis")
    for file in uploaded_files:
        with st.spinner(f"Analyzing {file.name}..."):
            features = analyze_audio(file.read(), file.name, analyze_full_song)
            audio_features.append(features)

            # Display basic info
            st.markdown(f"### 🎶 {file.name}")
            st.write(f"**Tempo:** {features['tempo']} BPM")
            st.write(f"**Key:** {features['key']}")
            st.write(f"**Mood:** {features['mood']}")
            st.write(f"**RMS:** {features['rms']}")
            st.write(f"**Spectral Centroid:** {features['spectral_centroid']}")
            st.write(f"**Spectral Bandwidth:** {features['spectral_bandwidth']}")
            st.write(f"**Spectral Contrast:** {features['spectral_contrast']}")
            st.write(f"**Zero Crossing Rate:** {features['zero_crossing_rate']}")
            st.audio(file)

            # Optional visualizations
            if st.checkbox(f"Show waveform for {file.name}"):
                fig, ax = plt.subplots()
                librosa.display.waveshow(features["y"], sr=features["sr"], ax=ax)
                ax.set_title("Waveform")
                st.pyplot(fig)

            if st.checkbox(f"Show spectrum for {file.name}"):
                D = np.abs(librosa.stft(features["y"]))**2
                freqs = librosa.fft_frequencies(sr=features["sr"])
                spectrum = D.mean(axis=1)
                fig2, ax2 = plt.subplots()
                ax2.semilogy(freqs, spectrum)
                ax2.set_xlabel('Frequency (Hz)')
                ax2.set_ylabel('Power')
                ax2.set_title('Frequency Spectrum')
                st.pyplot(fig2)

            if st.checkbox(f"Show energy over time for {file.name}"):
                rms_vals = librosa.feature.rms(y=features["y"])
                times = librosa.times_like(rms_vals, sr=features["sr"])
                fig3, ax3 = plt.subplots()
                ax3.plot(times, rms_vals[0])
                ax3.set_xlabel("Time (s)")
                ax3.set_ylabel("RMS Energy")
                ax3.set_title("Energy Over Time")
                st.pyplot(fig3)

            if st.checkbox(f"Show beat markers for {file.name}"):
                fig4, ax4 = plt.subplots()
                librosa.display.waveshow(features["y"], sr=features["sr"], ax=ax4)
                ax4.vlines(features["beats"] / features["sr"], -1, 1, color='r', alpha=0.5, linestyle='--')
                ax4.set_title("Beats Overlay")
                st.pyplot(fig4)

    # Playlist by mood
    st.subheader("🎧 Generate Playlist by Mood")
    moods = list(set(f["mood"] for f in audio_features))
    selected_mood = st.selectbox("Select mood:", moods)
    playlist = [f["name"] for f in audio_features if f["mood"] == selected_mood]
    st.write("**Playlist:**")
    for track in playlist:
        st.write(f"- {track}")

    # Export features to CSV
    st.subheader("📊 Export Audio Features")
    df = pd.DataFrame([{
        "File": f["name"],
        "Tempo (BPM)": f["tempo"],
        "Key": f["key"],
        "Mood": f["mood"],
        "RMS": f["rms"],
        "Spectral Centroid": f["spectral_centroid"],
        "Spectral Bandwidth": f["spectral_bandwidth"],
        "Spectral Contrast": f["spectral_contrast"],
        "Zero Crossing Rate": f["zero_crossing_rate"]
    } for f in audio_features])
    st.dataframe(df)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="pro_audio_features.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Upload one or more audio files to start analysis")
