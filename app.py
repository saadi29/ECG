import customtkinter as ctk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, iirnotch, resample, sosfiltfilt
from scipy.interpolate import make_interp_spline
import threading
import time
import serial
import serial.tools.list_ports
import os
import datetime
import re
from matplotlib.ticker import AutoMinorLocator

# --- ML IMPORTS ---
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURATION ---
BT_BAUDRATE = 460800  
RECORDING_DURATION = 10.0
PACKET_SIZE = 260    
DEFAULT_TRAIN_PATH = r"D:\OneDrive - Higher Education Commission\Roshaans Documents\GIKI\SEM7\EE441 DSP\CEP\mitbih_train.csv"
SAMPLING_RATE = 500  

# --- VISUAL THEME ---
COLOR_BG = "#121212"
COLOR_PANEL = "#1E1E1E"
COLOR_ACCENT = "#00ADB5"
COLOR_TEXT = "#EEEEEE"
COLOR_SUCCESS = "#00FF88"
COLOR_DANGER = "#FF2E63"
COLOR_WARNING = "#F39C12"

CLASS_MAP = {0: 'N (Normal)', 1: 'S (Supraventricular)', 2: 'V (Ventricular)', 3: 'F (Fusion)', 4: 'Q (Unknown)'}

# ==========================================
# 1. DSP ENGINE (ULTRA-EXTREME NO-RLD MODE)
# ==========================================
def conv_24bit(b):
    val = (b[0] << 16) | (b[1] << 8) | b[2]
    if val & 0x800000: val -= 0x1000000
    return val

def apply_filters(data, fs):
    if len(data) < fs: return data
    try:
        # OPTIMIZATION: Remove massive floating DC offset before filtering
        data = data - np.mean(data)
       
        nyquist = 0.5 * fs
       
        # 1. Wide "Trench" Notch Filters (Q=5.0) to catch massive uncompensated noise
        b_notch1, a_notch1 = iirnotch(50.0, 5.0, fs)
        data = filtfilt(b_notch1, a_notch1, data)

        # 2. First Harmonic
        b_notch2, a_notch2 = iirnotch(100.0, 5.0, fs)
        data = filtfilt(b_notch2, a_notch2, data)
       
        # 3. Second Harmonic
        b_notch3, a_notch3 = iirnotch(150.0, 5.0, fs)
        data = filtfilt(b_notch3, a_notch3, data)

        # 4. "Sniper" Bandpass Filter
        # OPTIMIZATION: Changed to 0.5 - 40.0 Hz.
        # 0.5 lets resting HR frequencies through without attenuation.
        # 40.0 keeps the QRS peak sharp so it stands out above the T-wave.
        low = 0.5 / nyquist
        high = 40.0 / nyquist
        sos = butter(4, [low, high], btype='band', output='sos')
       
        pad_len = min(int(fs*0.5), len(data)-10)
        data = sosfiltfilt(sos, data, padlen=pad_len)
        data = np.nan_to_num(data)
       
        return data
    except Exception as e:
        print(f"Filter Error: {e}")
        return data

def calculate_bpm(clean_sig, fs):
    try:
        if len(clean_sig) < fs or np.isnan(clean_sig).any(): return 0
       
        # Ignore filter transients at edges for threshold calculation
        trim = int(1.0 * fs)
        eval_sig = clean_sig[trim:-trim] if len(clean_sig) > 2 * trim else clean_sig
       
        # OPTIMIZATION: 0.3 * fs (300ms gap) steps over the T-wave. Max measurable HR is 200 BPM.
        distance = int(0.3 * fs)
        p5 = np.percentile(eval_sig, 5)
        p95 = np.percentile(eval_sig, 95)
       
        if p5 == p95: return 0
       
        thresh = p5 + (p95 - p5) * 0.6
        # OPTIMIZATION: Force peaks to actually protrude sharply (rejects rolling baseline wander)
        prominence_val = (p95 - p5) * 0.4
       
        raw_peaks, _ = find_peaks(clean_sig, distance=distance, height=thresh, prominence=prominence_val)
       
        # Discard false peaks detected in the transient regions
        peaks = [p for p in raw_peaks if trim <= p < len(clean_sig) - trim]
       
        if len(peaks) > 1:
            rr = np.diff(peaks) / fs
            bpm = int(60 / np.mean(rr))
            if 30 < bpm < 200: return bpm
        return 0
    except: return 0

def segment_beats_for_ml(sig, fs, target_len=187):
    if len(sig) == 0 or np.isnan(sig).any(): return np.array([]), []
       
    # Ignore filter transients at edges for threshold calculation
    trim = int(1.0 * fs)
    eval_sig = sig[trim:-trim] if len(sig) > 2 * trim else sig
   
    p5 = np.percentile(eval_sig, 5)
    p95 = np.percentile(eval_sig, 95)
   
    if p5 == p95: return np.array([]), []
   
    thresh = p5 + (p95 - p5) * 0.6
    prominence_val = (p95 - p5) * 0.4
   
    raw_peaks, _ = find_peaks(sig, distance=int(0.3*fs), height=thresh, prominence=prominence_val)
   
    # Discard false peaks detected in the transient regions
    peaks = [p for p in raw_peaks if trim <= p < len(sig) - trim]
   
    beats, valid_peaks = [], []
    window_s = 1.5
    window_samples = int(window_s * fs)
    pre_peak = int(0.3 * window_samples)
    post_peak = window_samples - pre_peak

    for p in peaks:
        start, end = p - pre_peak, p + post_peak
        if start >= 0 and end < len(sig):
            beat = sig[start:end]
            b_min, b_max = np.min(beat), np.max(beat)
            if b_max > b_min: beat = (beat - b_min) / (b_max - b_min)
            else: beat = np.zeros(len(beat))
           
            beat = resample(beat, target_len)
            beats.append(beat)
            valid_peaks.append(p)
           
    return np.array(beats), valid_peaks

# ==========================================
# 2. GUI & REPORT GENERATOR
# ==========================================
class ECGApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("A.I.D - Automated Intelligent Diagnosis")
        self.geometry("1400x900")
       
        self.active_port = None
        self.is_recording = False
        self.rf_model = None
        self.is_model_trained = False
        self.current_patient_name = "Unknown_Patient"
       
        self.grid_columnconfigure(1, weight=3)
        self.grid_columnconfigure(2, weight=1)
        self.setup_sidebar()
        self.setup_main_area()
        self.setup_right_panel()

    def setup_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=280, fg_color=COLOR_PANEL)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
       
        ctk.CTkLabel(self.sidebar, text="A.I.D ", font=("Arial", 24, "bold"), text_color=COLOR_ACCENT).pack(pady=20)
        self.combo_ports = ctk.CTkComboBox(self.sidebar, values=self.get_serial_ports())
        self.combo_ports.pack(padx=20, pady=5)
       
        self.btn_connect = ctk.CTkButton(self.sidebar, text="Select Hardware", command=self.toggle_serial)
        self.btn_connect.pack(padx=20, pady=5)
       
        ctk.CTkLabel(self.sidebar, text="PATIENT INFO", font=("Arial", 12, "bold"), text_color="gray").pack(pady=(20, 5))
        self.ent_patient = ctk.CTkEntry(self.sidebar, placeholder_text="e.g. Roshaan Wasif", fg_color="#2B2B2B", border_color="#333333")
        self.ent_patient.pack(padx=20, pady=5)
       
        ctk.CTkLabel(self.sidebar, text="ML CONFIGURATION", font=("Arial", 12, "bold"), text_color="gray").pack(pady=(20, 5))
        self.btn_train = ctk.CTkButton(self.sidebar, text="TRAIN MODEL", fg_color=COLOR_WARNING, text_color="black", command=self.train_model)
        self.btn_train.pack(padx=20, pady=10)
        self.lbl_train_status = ctk.CTkLabel(self.sidebar, text="Model: Not Trained", text_color=COLOR_DANGER, font=("Arial", 11))
        self.lbl_train_status.pack(pady=0)

        ctk.CTkLabel(self.sidebar, text="LIVE ANALYSIS", font=("Arial", 12, "bold"), text_color="gray").pack(pady=(20, 5))
        self.btn_record = ctk.CTkButton(self.sidebar, text="RECORD & CLASSIFY", fg_color="#27AE60", state="disabled", command=self.start_live_recording)
        self.btn_record.pack(padx=20, pady=10)
       
        self.frm_vitals = ctk.CTkFrame(self.sidebar, fg_color="#2B2B2B")
        self.frm_vitals.pack(fill="x", padx=20, pady=20)
        self.lbl_bpm = ctk.CTkLabel(self.frm_vitals, text="--", font=("Consolas", 40, "bold"), text_color=COLOR_ACCENT)
        self.lbl_bpm.pack(pady=10)
        ctk.CTkLabel(self.frm_vitals, text="BPM", font=("Arial", 10), text_color="gray").pack()
       
        self.status = ctk.CTkLabel(self.sidebar, text="System Ready", text_color="gray")
        self.status.pack(side="bottom", pady=20)

    def setup_main_area(self):
        self.main_area = ctk.CTkFrame(self, fg_color=COLOR_BG)
        self.main_area.grid(row=0, column=1, sticky="nsew")
        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 8))
        self.fig.patch.set_facecolor(COLOR_BG)
       
        titles = ["Lead I", "Lead II (ML Source)", "Lead III"]
        for i, ax in enumerate(self.axs):
            ax.set_facecolor("#000000")
            ax.grid(True, which='major', color='#333333', linewidth=0.8, alpha=0.5)
            ax.grid(True, which='minor', color='#111111', linewidth=0.4, alpha=0.3)
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
           
            ax.set_title(titles[i], color="white", loc="left", fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#333333')
            ax.spines['left'].set_color('#333333')
            ax.tick_params(colors='gray', labelsize=8)
       
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_area)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=20)

    def setup_right_panel(self):
        self.right_panel = ctk.CTkFrame(self, fg_color=COLOR_PANEL, width=300)
        self.right_panel.grid(row=0, column=2, sticky="nsew")
        self.card_result = ctk.CTkFrame(self.right_panel, fg_color="#2B2B2B", corner_radius=15)
        self.card_result.pack(fill="x", padx=20, pady=30)
        ctk.CTkLabel(self.card_result, text="ML DIAGNOSIS", font=("Arial", 12, "bold"), text_color="gray").pack(pady=(15,0))
        self.lbl_diag_main = ctk.CTkLabel(self.card_result, text="WAITING", font=("Arial", 24, "bold"), text_color="white", wraplength=200)
        self.lbl_diag_main.pack(pady=(5,15))
        self.lbl_details = ctk.CTkLabel(self.right_panel, text="--", font=("Consolas", 12), text_color="white")
        self.lbl_details.pack(pady=10)

    def train_model(self):
        if os.path.exists(DEFAULT_TRAIN_PATH):
            filepath = DEFAULT_TRAIN_PATH
        else:
            filepath = filedialog.askopenfilename(title="Select mitbih_train.csv")
        if not filepath: return
        self.btn_train.configure(state="disabled", text="TRAINING...")
        self.status.configure(text="Training RF Model...", text_color=COLOR_WARNING)
        self.update_idletasks()
        threading.Thread(target=self._train_thread, args=(filepath,), daemon=True).start()

    def _train_thread(self, fp):
        try:
            print("DEBUG: Loading CSV (Limit 10k rows)...")
            df = pd.read_csv(fp, header=None, nrows=10000)
            print(f"DEBUG: Loaded {len(df)} rows. Training...")
            self.rf_model = RandomForestClassifier(n_estimators=20, max_depth=10, n_jobs=-1, random_state=42)
            self.rf_model.fit(df.iloc[:,:-1].values, df.iloc[:,-1].values)
            self.is_model_trained = True
            self.after(0, lambda: self.training_complete(True))
        except Exception as e:
            print(f"ERROR: {e}")
            self.after(0, lambda: self.training_complete(False))

    def training_complete(self, s):
        self.btn_train.configure(state="normal", text="RETRAIN")
        if s: self.lbl_train_status.configure(text="Active", text_color=COLOR_SUCCESS)
        else: self.lbl_train_status.configure(text="Error", text_color=COLOR_DANGER)
        self.status.configure(text="Ready", text_color="gray")

    def get_serial_ports(self): return [p.device for p in serial.tools.list_ports.comports()] or ["No Ports"]
   
    def toggle_serial(self):
        if self.active_port is None:
            self.active_port = self.combo_ports.get()
            self.btn_connect.configure(text="Hardware Selected", fg_color=COLOR_SUCCESS)
            self.btn_record.configure(state="normal")
            self.status.configure(text="Ready to Record", text_color=COLOR_SUCCESS)
        else:
            self.active_port = None
            self.btn_connect.configure(text="Select Hardware", fg_color="#333333")
            self.btn_record.configure(state="disabled")

    def start_live_recording(self):
        self.btn_record.configure(text="WAITING...", state="disabled")
       
        raw_name = self.ent_patient.get().strip()
        self.current_patient_name = raw_name if raw_name else "Unknown_Patient"
       
        self._countdown(3)

    def _countdown(self, count):
        if count > 0:
            self.status.configure(text=f"Recording in {count}s...", text_color=COLOR_WARNING)
            self.after(1000, lambda: self._countdown(count - 1))
        else:
            self.is_recording = True
            self.btn_record.configure(text="RECORDING...")
            self.status.configure(text="Acquiring 10s...", text_color=COLOR_WARNING)
            threading.Thread(target=self._data_worker, daemon=True).start()

    def _data_worker(self):
        l1_raw, l2_raw = [], []
        buffer = bytearray()
        total_bytes = 0
        headers_found = 0
       
        SYNC_WORD = b'\x5A\x5A\xA5\xA5\x01\x05\x00'
       
        try:
            with serial.Serial(self.active_port, BT_BAUDRATE, timeout=1.0) as ser:
                ser.set_buffer_size(rx_size=1048576)
                ser.reset_input_buffer()
                start_t = time.time()
               
                while time.time() - start_t < RECORDING_DURATION:
                    bytes_waiting = ser.in_waiting
                    if bytes_waiting > 0:
                        new_data = ser.read(bytes_waiting)
                        buffer.extend(new_data)
                        total_bytes += len(new_data)
                   
                    while len(buffer) >= PACKET_SIZE:
                        idx = buffer.find(SYNC_WORD)
                       
                        if idx == -1:
                            if len(buffer) > 7:
                                buffer = buffer[-7:]
                            break
                           
                        if len(buffer) - idx < PACKET_SIZE:
                            break
                           
                        headers_found += 1
                       
                        packet = buffer[idx : idx + PACKET_SIZE]
                        buffer = buffer[idx + PACKET_SIZE :]
                       
                        data_chunk = packet[18:258]
                       
                        for s in range(10):
                            offset = s * 24
                            val1 = conv_24bit(data_chunk[offset:offset+3])
                            val2 = conv_24bit(data_chunk[offset+3:offset+6])
                            l1_raw.append(val1)
                            l2_raw.append(val2)
                   
                    time.sleep(0.002)

            if len(l1_raw) < 200:
                self.after(0, lambda: self.status.configure(text=f"No Data (B:{total_bytes} H:{headers_found} P:{len(l1_raw)//10})", text_color=COLOR_DANGER))
                self.after(0, lambda: self.btn_record.configure(text="RECORD", state="normal"))
                return

            self.after(0, lambda: self.status.configure(text="Processing...", text_color=COLOR_ACCENT))
           
            l1_np, l2_np = np.array(l1_raw), np.array(l2_raw)
            l3_np = l2_np - l1_np

            l1_cl = apply_filters(l1_np, SAMPLING_RATE)
            l2_cl = apply_filters(l2_np, SAMPLING_RATE)
            l3_cl = apply_filters(l3_np, SAMPLING_RATE)

            bpm = calculate_bpm(l2_cl, SAMPLING_RATE)

            diag_text, diag_color, peak_indices = "NORMAL", COLOR_SUCCESS, []
            detail = "No Model"
            beat_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
           
            if self.is_model_trained:
                beats, peak_indices = segment_beats_for_ml(l2_cl, SAMPLING_RATE)
                if len(beats) > 0:
                    preds = self.rf_model.predict(beats)
                    for p in preds:
                        beat_counts[int(p)] += 1
                   
                    if np.sum(preds > 0) > 0:
                        diag_text, diag_color = "ARRHYTHMIA", COLOR_DANGER
                       
                    detail = f"Normal: {beat_counts[0]} | Abnormal: {sum(beat_counts.values()) - beat_counts[0]}"
           
            results = {
                "l1": l1_cl, "l2": l2_cl, "l3": l3_cl,
                "bpm": bpm, "diag": diag_text, "clr": diag_color,
                "det": detail, "pks": peak_indices, "counts": beat_counts
            }
           
            self.after(0, self.update_gui, results)
            self.after(0, lambda: self.status.configure(text="Generating PDF Report...", text_color=COLOR_WARNING))
            threading.Thread(target=self.generate_pdf_report, args=(results,), daemon=True).start()

        except Exception as e:
            print(e)
            self.is_recording = False
            self.after(0, lambda: self.btn_record.configure(text="RECORD", state="normal"))
            self.after(0, lambda: self.status.configure(text=f"Error: {str(e)[:25]}", text_color=COLOR_DANGER))

    def generate_pdf_report(self, res):
        try:
            fig_pdf = plt.figure(figsize=(8.27, 11.69))
            fig_pdf.patch.set_facecolor('white')
           
            gs = fig_pdf.add_gridspec(5, 1, height_ratios=[1.5, 1.5, 1.5, 1.0, 1.0], hspace=0.5)
           
            axs_pdf = [fig_pdf.add_subplot(gs[0]), fig_pdf.add_subplot(gs[1]), fig_pdf.add_subplot(gs[2])]
            ax_text = fig_pdf.add_subplot(gs[3])
            ax_table = fig_pdf.add_subplot(gs[4])
           
            ax_text.axis('off')
            ax_table.axis('off')
           
            data = [res['l1'], res['l2'], res['l3']]
            titles = ["Lead I", "Lead II", "Lead III"]
           
            for i in range(3):
                ax = axs_pdf[i]
                sig = data[i]
                time_axis = np.arange(len(sig)) / SAMPLING_RATE
               
                ax.plot(time_axis, sig, color='black', linewidth=0.8)
                ax.set_title(f"{titles[i]} (Full 10-Second Trace)", loc="left", fontsize=10, fontweight='bold')
                ax.set_xlabel("Time (Seconds)", fontsize=8)
                ax.set_ylabel("Amplitude (\u03bcV)", fontsize=8)
               
                ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                ax.grid(True, which='major', color='#FFB6C1', linewidth=1.0)
                ax.grid(True, which='minor', color='#FFB6C1', linewidth=0.4, linestyle=':')
               
                ax.set_xlim(0, RECORDING_DURATION)
               
                if len(sig) > 0:
                    # Ignore the filter transients at the edges for Y-axis scaling
                    trim = int(1.0 * SAMPLING_RATE)
                    scale_sig = sig[trim:-trim] if len(sig) > 2 * trim else sig
                   
                    margin = (np.max(scale_sig) - np.min(scale_sig)) * 0.15
                    if margin == 0 or np.isnan(margin): margin = 1.0
                    ax.set_ylim(np.min(scale_sig) - margin, np.max(scale_sig) + margin)

            peaks = res['pks']
            if peaks:
                peak_times = [p / SAMPLING_RATE for p in peaks]
                axs_pdf[1].scatter(peak_times, [res['l2'][p] for p in peaks], color='red', marker='v', s=40, zorder=5)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
           
            report_text = (
                f"A.I.D. CLINICAL ECG REPORT\n"
                f"------------------------------------------------------\n"
                f"Patient Name: {self.current_patient_name}\n"
                f"Date & Time: {timestamp}\n"
                f"Sampling Rate: {SAMPLING_RATE} Hz | Duration: {RECORDING_DURATION}s\n"
                f"Heart Rate: {res['bpm']} BPM\n\n"
                f"SYSTEM DIAGNOSIS: {res['diag']}\n"
            )
           
            ax_text.text(0.05, 0.95, report_text, fontsize=11, va='top', ha='left', family='monospace')
           
            counts = res['counts']
            table_data = [
                ["Class Code", "Beat Type", "Count"],
                ["N", "Normal Beat", counts[0]],
                ["S", "Supraventricular Premature", counts[1]],
                ["V", "Premature Ventricular Contraction", counts[2]],
                ["F", "Fusion of Ventricular & Normal", counts[3]],
                ["Q", "Unclassifiable Beat", counts[4]]
            ]
           
            table = ax_table.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.15, 0.5, 0.15])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
           
            table.scale(1, 1.2)
           
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor('#E0E0E0')
           
            report_dir = r"D:\OneDrive - Higher Education Commission\Roshaans Documents\GIKI\FYP\stm code\reports"
            os.makedirs(report_dir, exist_ok=True)
           
            safe_name = re.sub(r'[^A-Za-z0-9_]', '', self.current_patient_name.replace(" ", "_"))
            if not safe_name: safe_name = "Unknown"
           
            filename = f"{safe_name}_ECG_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            full_filepath = os.path.join(report_dir, filename)
           
            fig_pdf.savefig(full_filepath, bbox_inches='tight', dpi=300)
            plt.close(fig_pdf)
           
            self.after(0, lambda: self.status.configure(text=f"Report Saved to: {filename}", text_color=COLOR_SUCCESS))
           
        except Exception as e:
            print(f"PDF Generation Error: {e}")
            self.after(0, lambda: self.status.configure(text="PDF Generation Failed", text_color=COLOR_DANGER))

    def update_gui(self, res):
        self.btn_record.configure(text="RECORD & CLASSIFY", state="normal")
        self.status.configure(text="Done", text_color=COLOR_SUCCESS)
        self.lbl_bpm.configure(text=str(res['bpm']) if res['bpm']>0 else "--")
        self.lbl_diag_main.configure(text=res['diag'], text_color=res['clr'])
        self.lbl_details.configure(text=res['det'])

        data = [res['l1'], res['l2'], res['l3']]
        colors = [COLOR_SUCCESS, "#FFD700", "#FF69B4"]
        titles = ["Lead I", "Lead II (ML)", "Lead III"]
       
        for i in range(3):
            self.axs[i].clear()
            self.axs[i].set_facecolor("#000000")
           
            sig = data[i]
           
            center = len(sig) // 2
            view_rad = 625
            start, end = max(0, center-view_rad), min(len(sig), center+view_rad)
           
            view_sig = sig[start:end]
            view_x_time = np.arange(start, end) / SAMPLING_RATE
           
            self.axs[i].plot(view_x_time, view_sig, color=colors[i], linewidth=1.5)

            self.axs[i].set_title(f"{titles[i]} (2.5s View)", color="white", loc="left", fontsize=10)
            self.axs[i].set_xlabel("Time (Seconds)", color="gray", fontsize=8)
           
            self.axs[i].xaxis.set_minor_locator(AutoMinorLocator(5))
            self.axs[i].yaxis.set_minor_locator(AutoMinorLocator(5))
            self.axs[i].grid(True, which='major', color='#3A3A3A', linewidth=0.9, alpha=0.8)
            self.axs[i].grid(True, which='minor', color='#1A1A1A', linewidth=0.5, alpha=0.5, linestyle='-')
           
            if len(view_sig) > 0:
                actual_max = np.max(view_sig)
                actual_min = np.min(view_sig)
                margin = (actual_max - actual_min) * 0.15
               
                if margin == 0 or np.isnan(margin):
                    margin = 1.0
                   
                self.axs[i].set_ylim(actual_min - margin, actual_max + margin)

        peaks = [p for p in res['pks'] if start <= p < end]
        if peaks:
            peak_times = [p / SAMPLING_RATE for p in peaks]
            self.axs[1].scatter(peak_times, [res['l2'][p] for p in peaks], color='red', marker='x', s=80, zorder=5)
       
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    app = ECGApp()
    app.mainloop()
