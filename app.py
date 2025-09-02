import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import tempfile
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# ----------------- Spotify API Setup -----------------
SPOTIFY_CLIENT_ID = st.secrets["SPOTIFY_CLIENT_ID"]
SPOTIFY_CLIENT_SECRET = st.secrets["SPOTIFY_CLIENT_SECRET"]

sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET
    )
)

# ----------------- Helper Functions -----------------
@st.cache_data
def analyze_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1)

        return {
            "tempo": round(tempo, 2),
            "spectral_centroid_mean": round(np.mean(spectral_centroids), 2),
            "spectral_rolloff_mean": round(np.mean(spectral_rolloff), 2),
            "zero_crossing_rate": round(zcr, 4),
            "mfccs_mean": [round(val, 2) for val in mfccs[:5]]
        }
    except Exception as e:
        return {"error": str(e)}

@st.cache_data
def get_song_info(song_name):
    try:
        results = sp.search(q=song_name, type="track", limit=1)
        if results["tracks"]["items"]:
            track = results["tracks"]["items"][0]
            return {
                "name": track["name"],
                "artist": track["artists"][0]["name"],
                "album": track["album"]["name"],
                "release_date": track["album"]["release_date"],
                "popularity": track["popularity"]
            }
        else:
            return {"error": "Song not found on Spotify."}
    except Exception as e:
        return {"error": str(e)}

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Music Analyzer Pro", layout="wide")
st.title("🎶 Ultimate Music Analyzer by Baibhav Ghimire")
st.write("Upload a song to analyze its features and get Spotify details.")

uploaded_file = st.file_uploader("Upload an MP3 or WAV file", type=["mp3", "wav"])

if uploaded_file is not None:
    with st.spinner("Processing audio..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            data, samplerate = sf.read(uploaded_file)
            sf.write(tmp_file.name, data, samplerate)
            temp_file_path = tmp_file.name

        analysis = analyze_audio(temp_file_path)

    if "error" in analysis:
        st.error(f"Audio Analysis Failed: {analysis['error']}")
    else:
        tab1, tab2 = st.tabs(["📊 Audio Features", "🎵 Spotify Info"])

        with tab1:
            st.subheader("Audio Features")
            st.json(analysis)

            y, sr = librosa.load(temp_file_path, sr=None)
            fig, ax = plt.subplots()
            librosa.display.waveshow(y, sr=sr, ax=ax)
            ax.set(title="Waveform")
            st.pyplot(fig)

        with tab2:
            song_query = st.text_input("Enter song name for Spotify info", value=os.path.splitext(uploaded_file.name)[0])
            if song_query:
                song_info = get_song_info(song_query)
                if "error" in song_info:
                    st.warning(song_info["error"])
                else:
                    st.write(f"**Song:** {song_info['name']}")
                    st.write(f"**Artist:** {song_info['artist']}")
                    st.write(f"**Album:** {song_info['album']}")
                    st.write(f"**Release Date:** {song_info['release_date']}")
                    st.write(f"**Popularity (Worldwide Reach):** {song_info['popularity']} / 100")
