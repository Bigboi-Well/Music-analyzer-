🎵 Pro Music Analyzer

Pro Music Analyzer is a professional-level music analysis tool built with Python and Streamlit. This project was developed as a group assignment under the supervision of a professor during a hands-on workshop on audio and AI technologies.

It allows users to upload audio files and get a detailed analysis including tempo, key, spectral features, energy, beats, mood, and more. Users can also generate playlists based on mood and export analysis results for further use.
---
🔹 Dependencies

streamlit — interactive web app framework

librosa — audio analysis library

numpy — numerical operations

matplotlib — plotting library

pandas — data management and CSV export

scipy — signal processing

requirements.txt example:

streamlit
librosa
numpy
matplotlib
pandas
scipy
---

🔹 How to Run

Launch the app:

streamlit run app.py


Open the app in your browser. Usually: http://localhost:8501.

Upload one or more audio files (mp3, wav, ogg).

Select analysis mode:[requirements.txt](https://github.com/user-attachments/files/22308647/requirements.txt)
streamlit
librosa
matplotlib
numpy
scipy
spotipy
soundfile

First 30 seconds: Fast preview

Full song: Detailed analysis

View audio features and visualizations: waveform, spectrum, energy over time, and beats.

Generate playlists by mood.

Export all features to CSV for further analysis.
---
