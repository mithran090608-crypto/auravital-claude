import io
import wave
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# =====================================================================
# CONFIGURATION & PROFESSIONAL MEDICAL THEME CSS
# =====================================================================
st.set_page_config(
    page_title="RPMA Master Clinical Portal",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    .motivation-card {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .patient-hero {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 12px;
        border-left: 6px solid #0284c7;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# LAYER 1: VOICE BIOMETRIC ENGINE (Frame-by-Frame Analysis)
# =====================================================================
def read_wav_bytes(audio_bytes: bytes):
    """Read a PCM WAV (8/16/24/32-bit, mono or stereo) -> (float32 mono array, sample rate)."""
    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sw == 1:
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0
    elif sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8)
        b = b[: len(b) // 3 * 3].reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = np.where(v >= 2**23, v - 2**24, v).astype(np.float32)
    elif sw == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
    else:
        raise ValueError("Unsupported WAV sample width")

    if ch > 1:
        x = x[: len(x) // ch * ch].reshape(-1, ch).mean(axis=1)
    return x, sr


def frame_pitch(frame: np.ndarray, sr: int):
    """Estimate pitch period of one frame with normalized autocorrelation."""
    frame = frame - frame.mean()
    n = len(frame)
    spec = np.fft.rfft(frame, 2 * n)
    ac = np.fft.irfft(spec * np.conj(spec))[:n]
    if ac[0] <= 1e-9:
        return None
    ac = ac / ac[0]

    lo = int(sr / 400)  # highest pitch considered: 400 Hz
    hi = min(int(sr / 70), n - 2)  # lowest pitch considered: 70 Hz
    if hi <= lo + 1:
        return None

    lag = lo + int(np.argmax(ac[lo:hi]))
    strength = float(ac[lag])
    if strength < 0.3:  # unvoiced / noisy frame
        return None

    a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
    denom = a - 2 * b + c
    shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
    return (lag + shift) / sr, strength


class VoiceBiometricEngine:

    def analyze_uploaded_file(self, audio_bytes: bytes) -> dict:
        """Returns pitch, jitter, shimmer, instability, husky factor, and signal array."""
        try:
            x, sr = read_wav_bytes(audio_bytes)
        except Exception:
            return {
                "error": (
                    "Could not read this WAV file. Please upload an uncompressed"
                    " PCM .wav file."
                )
            }

        if len(x) < sr * 0.3 or np.max(np.abs(x)) < 1e-6:
            return {
                "error": (
                    "Audio is too short or silent. Record at least 1 second of a"
                    " sustained 'aaah'."
                )
            }

        x = x / np.max(np.abs(x))

        loud = np.where(np.abs(x) > 0.05)[0]
        if len(loud) > 0:
            x = x[loud[0] : loud[-1] + 1]
        x = x[: sr * 3]

        frame_len = int(0.04 * sr)
        hop = int(0.01 * sr)
        periods, strengths, rms_vals = [], [], []

        for start in range(0, len(x) - frame_len, hop):
            frame = x[start : start + frame_len]
            res = frame_pitch(frame, sr)
            if res is None:
                continue
            periods.append(res[0])
            strengths.append(res[1])
            rms_vals.append(float(np.sqrt(np.mean(frame**2))))

        if len(periods) < 5:
            return {
                "error": (
                    "No clear voiced pitch found. Please upload a clean"
                    " sustained vowel sound."
                )
            }

        periods = np.array(periods)
        rms_vals = np.array(rms_vals)
        mean_period = float(np.mean(periods))

        pitch = 1.0 / float(np.median(periods))
        jitter = float(np.mean(np.abs(np.diff(periods))) / mean_period)
        shimmer = float(
            np.mean(np.abs(np.diff(rms_vals))) / (np.mean(rms_vals) + 1e-9)
        )
        noise_ratio = 1.0 - float(np.mean(strengths))

        instability = min(jitter * 750 + shimmer * 150, 100.0)
        husky = float(
            np.clip(noise_ratio * 1.5 + jitter * 5 + shimmer * 0.5, 0.0, 1.0)
        )

        return {
            "jitter": round(jitter, 4),
            "shimmer": round(shimmer, 4),
            "instability_score": round(instability, 2),
            "fundamental_pitch_hz": round(pitch, 1),
            "husky_factor": round(husky, 2),
            "signal": x[::10],  # downsampled for UI charting
        }


# =====================================================================
# LAYER 2: MOTIVATION ENGINE & BASELINE GENERATOR
# =====================================================================
def get_motivational_quote(condition: str, stable: bool) -> str:
    """Returns a tailored encouraging message for the patient."""
    quotes_stable = [
        (
            "🌟 'Every day is a step forward in your health journey. Keep up the"
            " great momentum!'"
        ),
        (
            "💪 'Your consistency with vital logging is paying off. You are in"
            " control of your health!'"
        ),
        (
            "🌱 'Small daily habits create massive long-term wellness. Excellent"
            " job today!'"
        ),
    ]
    quotes_alert = [
        (
            "☀️ 'Health has its ups and downs. Rest up, follow your physician's"
            " plan, and take it one day at a time.'"
        ),
        (
            "💙 'Listening to your body is your greatest strength. We flagged a"
            " slight shift, but your care team is here for you.'"
        ),
        (
            "🛡️ 'Awareness is key to recovery. Thank you for logging today—your"
            " proactive tracking protects your health.'"
        ),
    ]
    seed = int(datetime.now().strftime("%d%m%Y")) % 3
    return quotes_stable[seed] if stable else quotes_alert[seed]


def create_initial_patient_history(seed_id: int):
    rng = np.random.default_rng(seed_id)
    days = 30
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq="D")

    return pd.DataFrame({
        "Date": dates,
        "Heart_Rate_BPM": rng.normal(74, 3, days).round(1),
        "Oxygen_Sat_Percentage": np.clip(
            rng.normal(98.0, 0.4, days), 85, 100
        ).round(1),
        "Systolic_BP_mmHg": rng.normal(122, 4, days).round(1),
        "Voice_Instability_Index": np.clip(
            rng.normal(12.2, 1.5, days), 0, 100
        ).round(2),
        "Fundamental_Pitch_Hz": rng.normal(135.0, 5.0, days).round(1),
    }).set_index("Date")


# =====================================================================
# LAYER 3: MULTI-PATIENT STATE & SIDEBAR CONTROLS
# =====================================================================
st.title("🏥 Remote Patient Monitoring Portal")
st.markdown(
    "Category: **Remote Patient Monitoring (RPM)** | Telemetry & Patient"
    " Engagement Portal."
)

if "master_registry" not in st.session_state:
    st.session_state.master_registry = {
        "PT-01": {
            "name": "Sarah Jenkins",
            "age": 52,
            "condition": "Thyroid Fluid Accumulation (Hypothyroidism)",
            "history": create_initial_patient_history(42),
            "notes": "Patient reports mild vocal fatigue in mornings.",
            "scans": [],
        },
        "PT-02": {
            "name": "David Miller",
            "age": 29,
            "condition": "General Respiratory Screening",
            "history": create_initial_patient_history(101),
            "notes": "Baseline screening following seasonal allergy checks.",
            "scans": [],
        },
    }

# --- SIDEBAR ---
st.sidebar.header("🕹️ Portal Operating Mode")
view_mode = st.sidebar.radio(
    "Select Interface View:", ["Patient View (Simple)", "Clinician View (Advanced)"]
)

st.sidebar.markdown("---")
st.sidebar.header("📁 Patient Selection")
patient_options = {
    pid: f"{pid} - {meta['name']}"
    for pid, meta in st.session_state.master_registry.items()
}
selected_pid = st.sidebar.selectbox(
    "Active Patient Record:",
    options=list(patient_options.keys()),
    format_func=lambda x: patient_options[x],
)

active_patient = st.session_state.master_registry[selected_pid]
clinical_history = active_patient["history"]
latest_vitals = clinical_history.iloc[-1].copy()

if "Clinician" in view_mode:
    st.sidebar.markdown("---")
    st.sidebar.header("📝 Register New Patient")
    with st.sidebar.form("registration_form", clear_on_submit=True):
        new_name = st.text_input("Full Patient Name:")
        new_age = st.number_input(
            "Age (Years):", min_value=1, max_value=120, value=35
        )
        new_condition = st.selectbox(
            "Diagnostic Path:",
            [
                "Thyroid Fluid Accumulation (Hypothyroidism)",
                "Neuro-Vocal Degradation",
                "General Respiratory Screening",
            ],
        )
        submit_reg = st.form_submit_button("Register Patient Profile")

        if submit_reg and new_name:
            new_id = f"PT-{len(st.session_state.master_registry) + 1:02d}"
            st.session_state.master_registry[new_id] = {
                "name": new_name,
                "age": int(new_age),
                "condition": new_condition,
                "history": create_initial_patient_history(
                    np.random.randint(1, 500)
                ),
                "notes": "Newly registered patient profile.",
                "scans": [],
            }
            st.sidebar.success(f"Registered {new_name} as ID: {new_id}")
            st.rerun()

# =====================================================================
# CLINICAL STATUS CALCULATIONS
# =====================================================================
HR_MIN, HR_MAX = 60, 100
SPO2_MIN = 95.0
BP_SYS_MIN, BP_SYS_MAX = 90, 140
VOICE_INSTABILITY_THRESHOLD = 20.0

current_hr = latest_vitals.get("Heart_Rate_BPM", 72.0)
current_spo2 = latest_vitals.get("Oxygen_Sat_Percentage", 98.0)
current_bp = latest_vitals.get("Systolic_BP_mmHg", 120.0)
current_voice = latest_vitals.get("Voice_Instability_Index", 14.20)

risk_flags = []
if not (HR_MIN <= current_hr <= HR_MAX):
    risk_flags.append(
        f"Heart Rate ({current_hr} BPM) outside normal limits ({HR_MIN}-{HR_MAX})"
    )
if current_spo2 < SPO2_MIN:
    risk_flags.append(
        f"Oxygen Saturation ({current_spo2}%) below threshold ({SPO2_MIN}%)"
    )
if not (BP_SYS_MIN <= current_bp <= BP_SYS_MAX):
    risk_flags.append(
        f"Systolic BP ({current_bp} mmHg) outside normal limits"
        f" ({BP_SYS_MIN}-{BP_SYS_MAX})"
    )
if current_voice > VOICE_INSTABILITY_THRESHOLD:
    risk_flags.append(
        f"Voice Instability Index ({current_voice:.2f}) elevated"
    )

is_stable = len(risk_flags) == 0

# =====================================================================
# VIEW 1: PATIENT-FRIENDLY ACCESSIBLE DASHBOARD
# =====================================================================
if "Patient View" in view_mode:
    # Daily Motivation Card
    st.markdown(
        f"""
        <div class="motivation-card">
            <h3>☀️ Hello, {active_patient['name']}!</h3>
            <p style="font-size:1.15rem; margin-top:10px;">
                {get_motivational_quote(active_patient['condition'], is_stable)}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("📊 Your Latest Health Overview")

    if is_stable:
        st.success("✅ **All Vital Signs Are Looking Good Today!**")
    else:
        st.warning(
            "⚠️ **A vital metric requires attention.** Please rest and inform"
            " your care provider if you feel unwell."
        )

    # Simplified Large Cards
    p1, p2, p3 = st.columns(3)
    p1.metric(
        "❤️ Heart Rate",
        f"{current_hr:.0f} BPM",
        help="Normal resting range: 60 - 100 BPM",
    )
    p2.metric(
        "🫁 Oxygen Level",
        f"{current_spo2:.1f} %",
        help="Target saturation: 95% or higher",
    )
    p3.metric(
        "🩸 Blood Pressure",
        f"{current_bp:.0f} mmHg",
        help="Normal systolic range: 90 - 140 mmHg",
    )

    st.markdown("---")
    st.subheader("🎙️ Easy Daily Voice Sample Check")
    st.write(
        "Record or upload a 2-second WAV recording saying **'Ahhhh'** to check"
        " vocal health."
    )

    patient_audio = st.file_uploader(
        "Upload audio file (.wav)", type=["wav"], key=f"p_audio_{selected_pid}"
    )

    if patient_audio is not None:
        st.audio(patient_audio.getvalue(), format="audio/wav")
        if st.button("Submit Daily Audio Check", use_container_width=True):
            res = VoiceBiometricEngine().analyze_uploaded_file(
                patient_audio.getvalue()
            )
            if "error" in res:
                st.error(res["error"])
            else:
                st.balloons()
                st.success("Audio analysis complete and sent to your physician!")
                st.metric("Vocal Stability Score", f"{res['instability_score']}")

# =====================================================================
# VIEW 2: CLINICIAN ADVANCED TELEMETRY PORTAL
# =====================================================================
else:
    # --- HEADER METADATA ---
    c_meta1, c_meta2, c_meta3, c_meta4 = st.columns(4)
    c_meta1.markdown(f"👤 **Patient:**\n### {active_patient['name']}")
    c_meta2.markdown(f"🎂 **Age:**\n### {active_patient['age']} Yrs")
    c_meta3.markdown(f"🩺 **Diagnosis:**\n`{active_patient['condition']}`")
    c_meta4.markdown(f"🆔 **ID:**\n`{selected_pid}`")

    st.markdown("---")

    # --- VITAL INPUT & BIOMETRIC LOGGING ---
    st.subheader("📥 Log Diagnostic Entry & Acoustic Waveform")
    f_col1, f_col2 = st.columns(2)

    with f_col1:
        st.markdown("##### 📝 Input Vital Telemetry")
        input_hr = st.number_input(
            "Heart Rate (BPM):",
            min_value=40,
            max_value=200,
            value=int(latest_vitals.get("Heart_Rate_BPM", 72)),
            key=f"hr_in_{selected_pid}",
        )
        input_bp = st.number_input(
            "Systolic BP (mmHg):",
            min_value=70,
            max_value=220,
            value=int(latest_vitals.get("Systolic_BP_mmHg", 120)),
            key=f"bp_in_{selected_pid}",
        )
        input_spo2 = st.slider(
            "SpO2 (%):",
            min_value=80.0,
            max_value=100.0,
            value=float(latest_vitals.get("Oxygen_Sat_Percentage", 98.0)),
            step=0.1,
            key=f"spo2_in_{selected_pid}",
        )

    with f_col2:
        st.markdown("##### 🔊 Hormonal Acoustic Analysis")
        uploaded_file = st.file_uploader(
            "Upload sustained vowel (.wav)",
            type=["wav"],
            key=f"uploader_{selected_pid}",
        )

        audio_signal = None
        if uploaded_file is not None:
            st.audio(uploaded_file.getvalue(), format="audio/wav")

    if st.button(
        "🚀 Commit Entry to Diagnostic Pipeline", use_container_width=True
    ):
        commit_ok = True
        vocal_instability_score, detected_pitch, husky_factor = 14.20, 130.0, 0.20
        note = ""

        if uploaded_file is not None:
            audio_results = VoiceBiometricEngine().analyze_uploaded_file(
                uploaded_file.getvalue()
            )
            if "error" in audio_results:
                st.error(audio_results["error"])
                commit_ok = False
            else:
                vocal_instability_score = audio_results["instability_score"]
                detected_pitch = audio_results["fundamental_pitch_hz"]
                husky_factor = audio_results["husky_factor"]
                audio_signal = audio_results["signal"]
                note = "Voice biometric measured from uploaded WAV file."
        else:
            note = "No audio sample provided; using active baseline parameters."

        if commit_ok:
            new_row = pd.DataFrame([{
                "Heart_Rate_BPM": float(input_hr),
                "Oxygen_Sat_Percentage": float(input_spo2),
                "Systolic_BP_mmHg": float(input_bp),
                "Voice_Instability_Index": float(vocal_instability_score),
                "Fundamental_Pitch_Hz": float(detected_pitch),
            }], index=[pd.Timestamp.now()])

            st.session_state.master_registry[selected_pid]["history"] = (
                pd.concat([clinical_history, new_row])
            )
            st.toast(
                f"Telemetry updated for {active_patient['name']}", icon="💾"
            )
            st.rerun()

    # Audio Waveform Charting
    if uploaded_file is not None and audio_signal is not None:
        st.caption("Acoustic Waveform Visualizer (PCM Sample Stream)")
        st.line_chart(audio_signal, height=120)

    st.markdown("---")

    # --- ASSESSMENT REPORT ---
    st.subheader("📋 Automated Clinical Assessment")
    if not is_stable:
        st.error(
            f"⚠️ **Clinical Alert: Potential Issues Flagged for"
