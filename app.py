# --- CRITICAL BUGFIX: ROBUST ACTIVE-MEMORY PATH RESOLUTION ---
import sys
import types
import os

if 'pkg_resources' not in sys.modules:
    mock_pkg = types.ModuleType('pkg_resources')
    def resource_filename(package, resource):
        if package in sys.modules:
            mod = sys.modules[package]
            if hasattr(mod, '__file__') and mod.__file__:
                base_dir = os.path.dirname(mod.__file__)
                if os.path.exists(os.path.join(base_dir, resource)):
                    return os.path.join(base_dir, resource)
        try:
            parts = package.split('.')
            if parts[0] in sys.modules:
                root_mod = sys.modules[parts[0]]
                if hasattr(root_mod, '__file__') and root_mod.__file__:
                    current_path = os.path.dirname(root_mod.__file__)
                    for part in parts[1:]:
                        test_path = os.path.join(current_path, part)
                        if os.path.isdir(test_path):
                            current_path = test_path
                        else:
                            break
                    return os.path.join(current_path, resource)
        except Exception:
            pass
        return resource
        
    mock_pkg.resource_filename = resource_filename
    sys.modules['pkg_resources'] = mock_pkg

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import soundfile as sf
import pandas as pd
import collections
import pickle
import io

# Set up page configuration
st.set_page_config(page_title="EE200: Audio Fingerprinting App", layout="wide")
st.title("🎵 Zapptain America - Audio Fingerprinting System")
st.markdown("### EE200 Course Project: Signals, Systems and Networks")

# --- GLOBAL CONFIGURATIONS ---
WINDOW_LENGTH = 2048
HOP_LENGTH = 512
SR_RATE = 22050

# --- HELPER SIGNAL PROCESSING FUNCTIONS ---
def get_constellation_map(audio_file):
    """Loads audio safely and extracts clean local peaks using dynamic percentile thresholding."""
    try:
        file_bytes = audio_file.read() if hasattr(audio_file, 'read') else audio_file
        if hasattr(audio_file, 'seek'):
            audio_file.seek(0)
        
        try:
            sr, y = wavfile.read(io.BytesIO(file_bytes))
        except Exception:
            y, sr = sf.read(io.BytesIO(file_bytes))
            
        if y.ndim > 1:
            y = np.mean(y, axis=1)
            
        if sr != SR_RATE:
            num_samples = int(len(y) * SR_RATE / sr)
            y = signal.resample(y, num_samples)
    except Exception as e:
        st.error(f"Failed to decode audio file codec: {e}")
        y = np.zeros(SR_RATE * 5)

    if len(y) == 0 or np.max(np.abs(y)) < 1e-4:
        return np.array([]), np.array([]), np.zeros((1025, 100))

    # Compute STFT
    frequencies, times, Zxx = signal.stft(y, fs=SR_RATE, nperseg=WINDOW_LENGTH, noverlap=WINDOW_LENGTH - HOP_LENGTH)
    stft_db = 20 * np.log10(np.abs(Zxx) + 1e-10)
    stft_db = stft_db - np.max(stft_db)
    
    # Isolate top 1.5% highest energy peaks to prevent visual density saturation
    peak_threshold = np.percentile(stft_db, 98.5)
    
    neighborhood_size = (20, 20)
    local_max = ndimage.maximum_filter(stft_db, size=neighborhood_size) == stft_db
    foreground = (stft_db > peak_threshold)
    detected_peaks = local_max & foreground
    
    freq_indices, time_indices = np.where(detected_peaks)
    return time_indices, freq_indices, stft_db

def generate_hashes(time_indices, freq_indices):
    """Pairs constellation peaks within target zones into distinct hashes."""
    hashes = []
    num_peaks = len(time_indices)
    
    if num_peaks > 2000:
        time_indices = time_indices[:2000]
        freq_indices = freq_indices[:2000]
        num_peaks = 2000
        
    peaks = sorted(zip(time_indices, freq_indices), key=lambda x: x[0])
    
    for i in range(num_peaks):
        t1, f1 = peaks[i]
        for j in range(i + 1, num_peaks):
            t2, f2 = peaks[j]
            dt = t2 - t1
            if dt > 30:
                break
            if abs(f2 - f1) <= 8:
                hash_key = (int(f1), int(f2), int(dt))
                t1_seconds = t1 * (HOP_LENGTH / SR_RATE)
                hashes.append((hash_key, t1_seconds))
    return hashes

# --- DATABASE MANAGEMENT CLASS ---
class AudioDatabase:
    def __init__(self):
        self.db = collections.defaultdict(list)
        self.indexed_songs = set()
        self.load_database()

    def load_database(self):
        """Loads database from file."""

        target_file = None
        if os.path.exists("fingerprint_db.pkl"):
            target_file = "fingerprint_db.pkl"
        elif os.path.exists("fingerprint_db"):
            target_file = "fingerprint_db"
    
        loaded_dict = {}
        debug_info = "No fingerprint database file found."
    
        if target_file:
            try:
                with open(target_file, "rb") as f:
                    data = pickle.load(f)
    
                debug_info = f"File found! Object Type: {type(data)}"
    
                if hasattr(data, "db"):
                    loaded_dict = data.db
                    if hasattr(data, "indexed_songs"):
                        self.indexed_songs = set(data.indexed_songs)
    
                elif isinstance(data, (tuple, list)):
                    debug_info += f" (Sequence length {len(data)})"
                    for element in data:
                        if isinstance(element, dict):
                            loaded_dict = element
                        elif isinstance(element, (list, set, tuple)):
                            self.indexed_songs = set(str(x) for x in element)
    
                elif isinstance(data, dict):
                    loaded_dict = data
                    debug_info += f" (Dictionary containing {len(data)} entries)"
    
                if isinstance(loaded_dict, dict) and len(loaded_dict) > 0:
                    first_key = next(iter(loaded_dict))
    
                    if isinstance(first_key, str):
                        for song_name in loaded_dict:
                            self.indexed_songs.add(str(song_name))
                    else:
                        for _, v in loaded_dict.items():
                            if isinstance(v, (list, tuple, np.ndarray)):
                                for item in v:
                                    if isinstance(item, (list, tuple)) and len(item) > 0:
                                        self.indexed_songs.add(str(item[0]))
                                    elif isinstance(item, str):
                                        self.indexed_songs.add(item)
    
            except Exception as e:
                debug_info = f"Failed to parse pickle: {e}"
    
        self.db = collections.defaultdict(list)
    
        if isinstance(loaded_dict, dict):
            for k, v in loaded_dict.items():
                if isinstance(v, list):
                    self.db[k] = v
                elif isinstance(v, (tuple, np.ndarray)):
                    self.db[k] = list(v)
                else:
                    self.db[k] = [v]
    
        st.sidebar.markdown("### 🔍 Database Diagnostics")
        st.sidebar.info(debug_info)

    def identify_query(self, query_bytes):
        """Identifies an uploaded query clip using offset histogram alignment."""
        t_idx, f_idx, stft_db = get_constellation_map(query_bytes)
        
        if len(t_idx) == 0:
            return "No Match (Silent/Invalid File)", 0, t_idx, f_idx, stft_db, []

        query_hashes = generate_hashes(t_idx, f_idx)
        
        matches_found = collections.defaultdict(list)
        for hash_key, q_time_sec in query_hashes:
            if hash_key in self.db:
                for match_item in self.db[hash_key]:
                    if isinstance(match_item, (list, tuple)) and len(match_item) >= 2:
                        db_song_name = str(match_item[0])
                        try:
                            db_time_sec = float(match_item[1])
                            offset = db_time_sec - float(q_time_sec)
                            matches_found[db_song_name].append(offset)
                        except (ValueError, TypeError):
                            continue
        
        best_song = "Unknown / No Match"
        max_alignment_score = 0
        best_offsets_list = []
        
        for song_name, offsets in matches_found.items():
            if len(offsets) == 0:
                continue
            try:
                counts, _ = np.histogram(offsets, bins=np.arange(min(offsets)-1, max(offsets)+1, 0.5))
                highest_bin_peak = np.max(counts)
                
                if highest_bin_peak > max_alignment_score:
                    max_alignment_score = highest_bin_peak
                    best_song = song_name
                    best_offsets_list = offsets
            except Exception:
                continue
                
        return best_song, max_alignment_score, t_idx, f_idx, stft_db, best_offsets_list

# Force a fresh database read on every single refresh to break the cache lock
if "audio_db" not in st.session_state:
    st.session_state.audio_db = AudioDatabase()

# --- SIDEBAR DATABASE CONTROL PANEL ---
st.sidebar.header("🗄️ Song Database Management")

if st.session_state.audio_db.indexed_songs:
    st.sidebar.success(f"✅ Loaded {len(st.session_state.audio_db.indexed_songs)} tracks from fingerprint file!")
    st.sidebar.markdown("#### Currently Indexed Tracks:")
    st.sidebar.write(", ".join(list(st.session_state.audio_db.indexed_songs)))

# --- MAIN MODE TABS ---
tab1, tab2 = st.tabs(["🎯 Single-Clip Identification Mode", "📂 Batch Processing Mode"])

# ================= TAB 1: SINGLE-CLIP MODE =================
with tab1:
    st.header("Single-Clip Visual Identifier")
    st.write("Upload a query sound byte snippet to display the signal intermediate steps.")
    
    uploaded_file = st.file_uploader("Upload Clip Snippet:", type=["mp3", "wav", "m4a"], key="single_uploader")
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        with st.spinner("Analyzing spectral fingerprints..."):
            prediction, score, t_idx, f_idx, spectrogram_db, offsets = st.session_state.audio_db.identify_query(uploaded_file)
        
        st.success(f"### Predicted Song: **{prediction}**")
        st.metric(label="Histogram Match Alignment Confidence Score", value=int(score))
        
        st.markdown("### Intermediate Analysis Steps Visualizations")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 1. Computed Spectrogram & Extracted Constellation")
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            
            total_frames = spectrogram_db.shape[1]
            total_time = total_frames * (HOP_LENGTH / SR_RATE)
            
            ax1.imshow(spectrogram_db, origin='lower', aspect='auto', cmap='magma', 
                       extent=[0, total_time, 0, SR_RATE / 2])
            
            if len(t_idx) > 0:
                ax1.scatter(t_idx * (HOP_LENGTH / SR_RATE), f_idx * (SR_RATE / WINDOW_LENGTH), 
                            color='cyan', s=6, alpha=0.7, label="Constellation Peaks")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Frequency (Hz)")
            ax1.set_ylim(0, SR_RATE / 2)
            ax1.legend()
            st.pyplot(fig1)
            
        with col2:
            st.markdown("#### 2. Time-Offset Match Alignment Histogram")
            if len(offsets) > 0:
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                ax2.hist(offsets, bins=50, color='crimson', edgecolor='black', alpha=0.7)
                ax2.set_title("Peak Matching Target Bins Alignment")
                ax2.set_xlabel("Relative Time Offset (Seconds)")
                ax2.set_ylabel("Match Counts / Bin Strength")
                ax2.grid(axis='y', linestyle='--', alpha=0.5)
                st.pyplot(fig2)
            else:
                st.info("No matching hashes were identified in the library to construct an alignment profile.")

# ================= TAB 2: BATCH MODE =================
with tab2:
    st.header("Automatic Batch Processing Exporter")
    st.write("Upload a set of multiple query clips to compile the required evaluation report.")
    
    batch_files = st.file_uploader("Upload Query Collection Batch Files:", type=["mp3", "wav", "m4a"], accept_multiple_files=True, key="batch_uploader")
    
    if batch_files:
        st.write(f"Loaded {len(batch_files)} clips for automatic processing.")
        
        if st.button("Run Batch Identification Analysis"):
            batch_records = []
            batch_progress = st.progress(0)
            
            for idx, file_obj in enumerate(batch_files):
                clean_filename = os.path.splitext(file_obj.name)[0]
                pred_song, _, _, _, _, _ = st.session_state.audio_db.identify_query(file_obj)
                
                batch_records.append({
                    "filename": clean_filename,
                    "prediction": pred_song
                })
                batch_progress.progress((idx + 1) / len(batch_files))
            
            results_df = pd.DataFrame(batch_records)
            st.markdown("### Previewing Analysis Output Results Table")
            st.dataframe(results_df, use_container_width=True)
            
            csv_buffer = results_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download compiled results.csv",
                data=csv_buffer,
                file_name="results.csv",
                mime="text/csv"
            )
            st.success("Batch classification complete! Download the CSV sheet above.")
