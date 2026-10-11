import io
import wave
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# --- Machine learning (scikit-learn) ---
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# =====================================================================
# CONFIGURATION & PROFESSIONAL MEDICAL THEME CSS
# =====================================================================
st.set_page_config(
    page_title="RPMA Master Clinical Panel",
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
    .clinical-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
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
        x = np.where(v >= 2 ** 23, v - 2 ** 24, v).astype(np.float32)
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

    lo = int(sr / 400)            # highest pitch considered: 400 Hz
    hi = min(int(sr / 70), n - 2)  # lowest pitch considered: 70 Hz
    if hi <= lo + 1:
        return None

    lag = lo + int(np.argmax(ac[lo:hi]))
    strength = float(ac[lag])
    if strength < 0.3:            # unvoiced / noisy frame
        return None

    a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
    denom = a - 2 * b + c
    shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
    return (lag + shift) / sr, strength


class VoiceBiometricEngine:
    def analyze_uploaded_file(self, audio_bytes: bytes) -> dict:
        """Returns pitch, jitter, shimmer, instability and husky factor,
        or {"error": "..."} if the audio can't be analysed."""
        try:
            x, sr = read_wav_bytes(audio_bytes)
        except Exception:
            return {"error": "Could not read this WAV file. Please upload an uncompressed PCM .wav (e.g. 16-bit)."}

        if len(x) < sr * 0.3 or np.max(np.abs(x)) < 1e-6:
            return {"error": "Audio is too short or silent. Record at least 1 second of a sustained 'aaah'."}

        x = x / np.max(np.abs(x))

        loud = np.where(np.abs(x) > 0.05)[0]
        if len(loud) > 0:
            x = x[loud[0]: loud[-1] + 1]
        x = x[: sr * 3]

        frame_len = int(0.04 * sr)
        hop = int(0.01 * sr)
        periods, strengths, rms_vals = [], [], []

        for start in range(0, len(x) - frame_len, hop):
            frame = x[start: start + frame_len]
            res = frame_pitch(frame, sr)
            if res is None:
                continue
            periods.append(res[0])
            strengths.append(res[1])
            rms_vals.append(float(np.sqrt(np.mean(frame ** 2))))

        if len(periods) < 5:
            return {"error": "No clear voiced pitch found. Please upload a clean sustained vowel sound."}

        periods = np.array(periods)
        rms_vals = np.array(rms_vals)
        mean_period = float(np.mean(periods))

        pitch = 1.0 / float(np.median(periods))
        jitter = float(np.mean(np.abs(np.diff(periods))) / mean_period)
        shimmer = float(np.mean(np.abs(np.diff(rms_vals))) / (np.mean(rms_vals) + 1e-9))
        noise_ratio = 1.0 - float(np.mean(strengths))

        instability = min(jitter * 750 + shimmer * 150, 100.0)
        husky = float(np.clip(noise_ratio * 1.5 + jitter * 5 + shimmer * 0.5, 0.0, 1.0))

        return {
            "jitter": round(jitter, 4),
            "shimmer": round(shimmer, 4),
            "instability_score": round(instability, 2),
            "fundamental_pitch_hz": round(pitch, 1),
            "husky_factor": round(husky, 2),
        }


# =====================================================================
# LAYER 2: BASELINE GENERATOR FOR NEW PATIENTS
# =====================================================================
def create_initial_patient_history(seed_id: int):
    rng = np.random.default_rng(seed_id)
    days = 30
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq="D")

    return pd.DataFrame({
        "Date": dates,
        "Heart_Rate_BPM": rng.normal(74, 3, days).round(1),
        "Oxygen_Sat_Percentage": np.clip(rng.normal(98.0, 0.4, days), 85, 100).round(1),
        "Systolic_BP_mmHg": rng.normal(122, 4, days).round(1),
        "Voice_Instability_Index": np.clip(rng.normal(12.2, 1.5, days), 0, 100).round(2),
        "Fundamental_Pitch_Hz": rng.normal(135.0, 5.0, days).round(1),
    }).set_index("Date")


# =====================================================================
# LAYER 2B: MACHINE LEARNING ENGINE (NEW)
#   1) Isolation Forest  -> anomaly detection (patient-specific baseline)
#   2) Linear Regression -> trend analysis + 7-step forecast
# =====================================================================
ML_FEATURES = [
    "Heart_Rate_BPM",
    "Oxygen_Sat_Percentage",
    "Systolic_BP_mmHg",
    "Voice_Instability_Index",
    "Fundamental_Pitch_Hz",
]
MIN_ROWS_FOR_ML = 10


def detect_anomalies(history: pd.DataFrame, contamination: float = 0.05):
    """Train an Isolation Forest on the patient's PAST entries (everything except
    the newest one), then score every entry. The newest entry is therefore
    judged as 'new data' against the patient's own learned baseline.

    Returns (scores_df, error_message). scores_df has columns:
    Anomaly_Score (lower = more abnormal) and Is_Anomaly (bool).
    """
    data = history[ML_FEATURES].dropna()
    if len(data) < MIN_ROWS_FOR_ML:
        return None, f"Need at least {MIN_ROWS_FOR_ML} entries for anomaly detection (have {len(data)})."

    train = data.iloc[:-1]
    scaler = StandardScaler().fit(train)
    model = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=42,
    ).fit(scaler.transform(train))

    scaled_all = scaler.transform(data)
    scores = model.decision_function(scaled_all)    # < 0 => outlier
    flags = model.predict(scaled_all) == -1

    result = pd.DataFrame(
        {"Anomaly_Score": scores.round(3), "Is_Anomaly": flags},
        index=data.index,
    )
    return result, None


def forecast_trend(series: pd.Series, horizon: int = 7, window: int = 30):
    """Fit a Linear Regression on the last `window` readings and project
    `horizon` steps ahead. Returns a dict with slope, direction, forecast, r2."""
    s = series.dropna().tail(window)
    if len(s) < MIN_ROWS_FOR_ML:
        return None

    X = np.arange(len(s)).reshape(-1, 1)
    y = s.values
    model = LinearRegression().fit(X, y)

    future_X = np.arange(len(s), len(s) + horizon).reshape(-1, 1)
    forecast = model.predict(future_X)

    slope = float(model.coef_[0])
    drift = abs(slope) * len(s)
    rel = drift / (abs(float(np.mean(y))) + 1e-9)
    if rel < 0.05:
        direction = "Stable"
    else:
        direction = "Rising" if slope > 0 else "Falling"

    return {
        "slope": slope,
        "direction": direction,
        "r2": float(model.score(X, y)),
        "forecast": forecast,
        "history_len": len(s),
        "history": y,
    }


# =====================================================================
# LAYER 3: APP UI & MULTI-PATIENT STATE
# =====================================================================
st.title("🏥 Remote Patient Monitoring Agent (RPMA)")
st.markdown("Category: **Remote Patient Monitoring (RPM)** | Enterprise Endocrine & Neuro-Respiratory Telemetry Framework.")
st.markdown("---")

if "master_registry" not in st.session_state:
    st.session_state.master_registry = {
        "PT-01": {
            "name": "Sarah Jenkins",
            "age": 52,
            "condition": "Thyroid Fluid Accumulation (Hypothyroidism)",
            "history": create_initial_patient_history(42),
            "notes": "Patient reports mild vocal fatigue in mornings."
        },
        "PT-02": {
            "name": "David Miller",
            "age": 29,
            "condition": "General Respiratory Screening",
            "history": create_initial_patient_history(101),
            "notes": "Baseline screening following seasonal allergy checks."
        },
    }

# --- SIDEBAR ---
st.sidebar.header("📝 Register New Patient")
with st.sidebar.form("registration_form", clear_on_submit=True):
    new_name = st.text_input("Full Patient Name:")
    new_age = st.number_input("Age (Years):", min_value=1, max_value=120, value=35)
    new_condition = st.selectbox("Suspected Target Diagnostics Condition:", [
        "Thyroid Fluid Accumulation (Hypothyroidism)",
        "Neuro-Vocal Degradation",
        "General Respiratory Screening",
    ])
    submit_registration = st.form_submit_button("Register Patient Profile")

    if submit_registration and new_name:
        new_id = f"PT-{len(st.session_state.master_registry) + 1:02d}"
        st.session_state.master_registry[new_id] = {
            "name": new_name,
            "age": int(new_age),
            "condition": new_condition,
            "history": create_initial_patient_history(np.random.randint(1, 500)),
            "notes": "Newly registered patient profile."
        }
        st.sidebar.success(f"Registered {
