# Music-analyzer-
🎵 Pro Music Analyzer

A professional-level music analysis tool built with Python and Streamlit. Upload your audio files (mp3, wav, ogg) and get a detailed analysis including tempo, key, spectral features, energy, beats, mood, and more. Generate playlists by mood and export features for further use.

🔹 Features

Tempo Detection: Beats per minute (BPM)

Key Detection: Musical key of the track

Mood Classification: Energetic, Calm, or Neutral based on RMS and spectral features

Spectral Features:

Spectral Centroid

Spectral Bandwidth

Spectral Contrast

Zero Crossing Rate

Waveform & Frequency Spectrum: Visualize the audio signal

Energy Over Time Plot: RMS energy variation across the track

Beat Overlay Visualization: Shows detected beats on waveform

Playlist Generation: Group songs by mood

Export Audio Features: CSV export for all analyzed features

Performance Optimizations:

Option to analyze only first 30 seconds (faster)

Optional full song analysis

Downsampling to speed up analysis

Streamlit caching to avoid reprocessing

🔹 Installation

Clone the repository:

git clone https://github.com/yourusername/pro-music-analyzer.git
cd pro-music-analyzer


Create a virtual environment (optional but recommended):

python -m venv venv


Activate the virtual environment:

Windows:

venv\Scripts\activate


Mac/Linux:

source venv/bin/activate


Install dependencies:

pip install -r requirements.txt

🔹 Dependencies

The main Python packages used are:

streamlit — for interactive web app

librosa — audio analysis

numpy — numerical operations

matplotlib — plotting waveforms & spectra

pandas — exporting data

scipy — signal processing

requirements.txt example:

streamlit
librosa
numpy
matplotlib
pandas
scipy

🔹 Usage

Run the app:

streamlit run app.py


Open in browser:
Streamlit usually opens the app automatically in your default browser. If not, check the terminal URL, usually: http://localhost:8501.

Upload audio files:
Supports .mp3, .wav, and .ogg.

Choose analysis mode:

First 30 seconds: Fast analysis

Full song: Full analysis (slower)

Visualize features:

Waveform

Frequency spectrum

Energy over time

Beat markers

Generate playlist by mood

Export audio features to CSV
