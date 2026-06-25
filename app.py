import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import pandas as pd
import collections
import os

# Set page configuration with a neat theme
st.set_page_config(page_title="EE200: Audio Fingerprinting App", layout="wide")
st.title("🎵 Zapptain America - Audio Fingerprinting System")
st.markdown("### EE200 Course Project: Signals, Systems and Networks")

# --- CORE SIGNAL PROCESSING ENGINE (From Q3A) ---

WINDOW_LENGTH = 2048
HOP_LENGTH = WINDOW_LENGTH // 4
SR_RATE = 22050

@st.cache_data
def get_constellation_map(audio_bytes):
    """
    Loads audio from memory bytes and extracts the local peak constellation map.
    Cached by Streamlit to keep the app highly responsive.
    """
    # Load audio directly from memory buffer
    y, sr = librosa.load(audio_bytes, sr=SR_RATE)
    stft_matrix = librosa.stft(y, n_fft=WINDOW_LENGTH, hop_length=HOP_LENGTH)
    stft_db = librosa.amplitude_to_db(np.abs(stft_matrix), ref=np.max)
    
    # 2D Max Filter for local peak detection
    neighborhood_size = (20, 20)
    local_max = ndimage.maximum_filter(stft_db, size=neighborhood_size) == stft_db
    foreground = (stft_db > -45)
    detected_peaks = local_max & foreground
    
    freq_indices, time_indices = np.where(detected_peaks)
    return time_indices, freq_indices, stft_db

def generate_hashes(time_indices, freq_indices):
    """Pairs constellation peaks within target zones into distinct hashes."""
    hashes = []
    num_peaks = len(time_indices)
    peaks = sorted(zip(time_indices, freq_indices), key=lambda x: x[0])
    
    for i in range(num_peaks):
        t1, f1 = peaks[i]
        for j in range(i + 1, num_peaks):
            t2, f2 = peaks[j]
            dt = t2 - t1
            if dt > 30:  # target zone max time gap constraint
                break
            if abs(f2 - f1) <= 8:  # target zone max frequency delta constraint
                hash_key = (int(f1), int(f2), int(dt))
                t1_seconds = t1 * (HOP_LENGTH / SR_RATE)
                hashes.append((hash_key, t1_seconds))
    return hashes

# --- DATABASE MANAGEMENT CLASS ---

# class AudioDatabase:
#     def __init__(self):
#         # Maps hash_key -> list of (song_name, t1_seconds)
#         self.db = collections.defaultdict(list)
#         self.indexed_songs = set()
    # --- DATABASE MANAGEMENT CLASS ---
# class AudioDatabase:
#     def __init__(self):
#         # Maps hash_key -> list of (song_name, t1_seconds)
#         self.db = collections.defaultdict(list)
#         self.indexed_songs = set()
#        # self.database_file = "fingerprint_db.pkl" # <-- Define the file name here

#     def index_song(self, audio_path, song_name):
# --- DATABASE MANAGEMENT CLASS ---
class AudioDatabase:
    def __init__(self):
        # Maps hash_key -> list of (song_name, t1_seconds)
        self.db = collections.defaultdict(list)
        self.indexed_songs = set()
        self.database_file = "fingerprint_db.pkl"  # <-- Make sure this is inside __init__
        self.load_database()                       # <-- Automatically load on startup

    def load_database(self):
        """Loads the fingerprint database from the pickle file if it exists."""
        import pickle
        import os
        if os.path.exists(self.database_file):
            try:
                with open(self.database_file, "rb") as f:
                    self.db = pickle.load(f)
                # Rebuild the indexed songs set based on loaded keys
                self.indexed_songs = set(
                    song for songs_list in self.db.values() for song, _ in songs_list
                )
            except Exception as e:
                st.error(f"Error loading database: {e}")

    def save_database(self):
        """Saves the current fingerprint database to the pickle file."""
        import pickle
        try:
            with open(self.database_file, "wb") as f:
                pickle.dump(self.db, f)
        except Exception as e:
            st.error(f"Error saving database: {e}")

   def index_song(self, audio_path, song_name):
        """Indexes a standard local disk song into the global database."""
        try:
            # Index middle section of the song database tracks to optimize space
            y, sr = librosa.load(audio_path, sr=SR_RATE, offset=30.0, duration=60.0)
            
            stft_matrix = librosa.stft(y, n_fft=WINDOW_LENGTH, hop_length=HOP_LENGTH)
            stft_db = librosa.amplitude_to_db(np.abs(stft_matrix), ref=np.max)
            
            neighborhood_size = (20, 20)
            local_max = ndimage.maximum_filter(stft_db, size=neighborhood_size) == stft_db
            foreground = (stft_db > -45)
            detected_peaks = local_max & foreground
            
            freq_indices, time_indices = np.where(detected_peaks)
            frame_offset = int(30.0 * SR_RATE / HOP_LENGTH)
            time_indices_adjusted = time_indices + frame_offset
            
            hashes = generate_hashes(time_indices_adjusted, freq_indices)
            for hash_key, t1_seconds in hashes:
                self.db[hash_key].append((song_name, t1_seconds))
            self.indexed_songs.add(song_name)
        except Exception as e:
            st.error(f"Error indexing {song_name}: {e}")
            self.save_database()

    def identify_query(self, query_bytes):
        """Identifies an uploaded query clip using offset histogram alignment."""
        t_idx, f_idx, stft_db = get_constellation_map(query_bytes)
        query_hashes = generate_hashes(t_idx, f_idx)
        
        matches_found = collections.defaultdict(list)
        for hash_key, q_time_sec in query_hashes:
            if hash_key in self.db:
                for db_song_name, db_time_sec in self.db[hash_key]:
                    offset = db_time_sec - q_time_sec
                    matches_found[db_song_name].append(offset)
        
        best_song = "Unknown / No Match"
        max_alignment_score = 0
        best_offsets_list = []
        
        for song_name, offsets in matches_found.items():
            if len(offsets) == 0:
                continue
            counts, _ = np.histogram(offsets, bins=np.arange(min(offsets)-1, max(offsets)+1, 0.5))
            highest_bin_peak = np.max(counts)
            
            if highest_bin_peak > max_alignment_score:
                max_alignment_score = highest_bin_peak
                best_song = song_name
                best_offsets_list = offsets
                
        return best_song, max_alignment_score, t_idx, f_idx, stft_db, best_offsets_list

# Initialize or restore database in Streamlit's operational session state
if 'audio_db' not in st.session_state:
    st.session_state.audio_db = AudioDatabase()

# --- SIDEBAR DATABASE CONTROL PANEL ---

st.sidebar.header("🗄️ Song Database Management")
db_folder = st.sidebar.text_input("Database Directory Path:", "EE200 Project Song Database")

if st.sidebar.button("Index / Reload Song Database"):
    if os.path.exists(db_folder):
        song_files = [f for f in os.listdir(db_folder) if f.endswith(('.mp3', '.wav', '.m4a'))]
        if song_files:
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
            
            # Reset database instances
            st.session_state.audio_db = AudioDatabase()
            
            for index, file_name in enumerate(song_files):
                status_text.text(f"Indexing: {file_name}")
                song_path = os.path.join(db_folder, file_name)
                label_name = os.path.splitext(file_name)[0]
                st.session_state.audio_db.index_song(song_path, label_name)
                progress_bar.progress((index + 1) / len(song_files))
                
            status_text.text(f"✅ Successfully Indexed {len(song_files)} songs!")
        else:
            st.sidebar.error("No valid audio files found in the directory.")
    else:
        st.sidebar.error("Directory path not found. Verify your storage location.")

# Show currently indexed tracks
if st.session_state.audio_db.indexed_songs:
    st.sidebar.markdown("#### Currently Indexed Tracks:")
    st.sidebar.write(", ".join(list(st.session_state.audio_db.indexed_songs)))
else:
    st.sidebar.warning("Database empty. Please set your folder path and index the library.")

# --- MAIN MODE TABS (Single-Clip vs Batch) ---

tab1, tab2 = st.tabs(["🎯 Single-Clip Identification Mode", "📂 Batch Processing Mode"])

# ================= TAB 1: SINGLE-CLIP MODE =================
with tab1:
    st.header("Single-Clip Visual Identifier")
    st.write("Upload a query sound byte snippet to display the signal intermediate steps.")
    
    uploaded_file = st.file_file = st.file_uploader("Upload Clip Snippet:", type=["mp3", "wav", "m4a"], key="single_uploader")
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        if not st.session_state.audio_db.indexed_songs:
            st.error("Cannot execute lookup: Database is not indexed yet. Use the sidebar panel.")
        else:
            with st.spinner("Analyzing spectral fingerprints..."):
                # Execute fingerprint analysis
                prediction, score, t_idx, f_idx, spectrogram_db, offsets = st.session_state.audio_db.identify_query(uploaded_file)
            
            # Show final diagnostic identification card
            st.success(f"### Predicted Song: **{prediction}**")
            st.metric(label="Histogram Match Alignment Confidence Score", value=int(score))
            
            # Render visual processing graphs
            st.markdown("### Intermediate Analysis Steps Visualizations")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 1. Computed Spectrogram & Extracted Constellation")
                fig1, ax1 = plt.subplots(figsize=(10, 5))
                librosa.display.specshow(spectrogram_db, sr=SR_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='linear', cmap='magma', ax=ax1)
                ax1.scatter(t_idx * (HOP_LENGTH / SR_RATE), f_idx * (SR_RATE / WINDOW_LENGTH), color='cyan', s=15, alpha=0.6, label="Constellation Peaks")
                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Frequency (Hz)")
                ax1.legend()
                st.pyplot(fig1)
                
            with col2:
                st.markdown("#### 2. Time-Offset Match Alignment Histogram")
                if len(offsets) > 0:
                    fig2, ax2 = plt.subplots(figsize=(10, 5))
                    ax2.hist(offsets, bins=50, color='crimson', edgecolor='black', alpha=0.7)
                    ax2.set_title(f"Peak Matching Target Bins Alignment")
                    ax2.set_xlabel("Relative Time Offset (Database Time - Query Time in Seconds)")
                    ax2.set_ylabel("Match Counts / Bin Strength")
                    ax2.grid(axis='y', linestyle='--', alpha=0.5)
                    st.pyplot(fig2)
                else:
                    st.info("No matching hashes were identified in the library to construct an alignment profile.")

# ================= TAB 2: BATCH MODE =================
with tab2:
    st.header("Automatic Batch Processing Exporter")
    st.write("Upload a set of multiple query clips to compile the required evaluation evaluation report.")
    
    batch_files = st.file_uploader("Upload Query Collection Batch Files:", type=["mp3", "wav", "m4a"], accept_multiple_files=True, key="batch_uploader")
    
    if batch_files:
        if not st.session_state.audio_db.indexed_songs:
            st.error("Cannot process batch: Index the library using the sidebar panel first.")
        else:
            st.write(f"Loaded {len(batch_files)} clips for automatic processing.")
            
            if st.button("Run Batch Identification Analysis"):
                batch_records = []
                batch_progress = st.progress(0)
                
                for idx, file_obj in enumerate(batch_files):
                    # Extract original name without the storage folder path extensions
                    clean_filename = os.path.splitext(file_obj.name)[0]
                    
                    # Run lookup
                    pred_song, _, _, _, _, _ = st.session_state.audio_db.identify_query(file_obj)
                    
                    batch_records.append({
                        "filename": clean_filename,
                        "prediction": pred_song
                    })
                    batch_progress.progress((idx + 1) / len(batch_files))
                
                # Convert results list array into Pandas DataFrame format
                results_df = pd.DataFrame(batch_records)
                
                st.markdown("### Previewing Analysis Output Results Table")
                st.dataframe(results_df, use_container_width=True)
                
                # Convert explicitly to the required results.csv format mapping schema 
                csv_buffer = results_df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="📥 Download compiled results.csv",
                    data=csv_buffer,
                    file_name="results.csv",
                    mime="text/csv"
                )
                st.success("Batch classification complete! Download the CSV sheet above.")
