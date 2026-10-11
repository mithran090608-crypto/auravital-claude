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
    .alert-box {
        background-color: #fff1f2;
        border-left: 5px solid #e11d48;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    .success-box {
        background-color: #f0fdf4;
        border-left: 5px solid #16a34a;
        padding: 15px;
        border-radius: 4px;
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
        or fallback baseline metrics if audio upload is missing."""
        if not audio_bytes:
            return {
                "fundamental_pitch_hz": 153.4,
                "jitter": 0.021,
                "shimmer": 0.045,
                "instability_score": 46.22,
                "husky_factor": 0.80,
            }
        try:
            x, sr = read_wav_bytes(audio_bytes)
        except Exception:
            return {
                "fundamental_pitch_hz": 153.4,
                "jitter": 0.021,
                "shimmer": 0.045,
                "instability_score": 46.22,
                "husky_factor": 0.80,
            }

        if len(x) < sr * 0.3 or np.max(np.abs(x)) < 1e-6:
            return {
                "fundamental_pitch_hz": 145.0,
                "jitter": 0.015,
                "shimmer": 0.030,
                "instability_score": 25.0,
                "husky_factor": 0.45,
            }

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
            return {
                "fundamental_pitch_hz": 150.0,
                "jitter": 0.018,
                "shimmer": 0.035,
                "instability_score": 30.0,
                "husky_factor": 0.50,
            }

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
# LAYER 2: BASELINE GENERATOR FOR PATIENT TELEMETRY
# =====================================================================
def create_initial_patient_history(seed_id: int):
    rng = np.random.default_rng(seed_id)
    days = 31
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq="D")

    return pd.DataFrame({
        "Date": dates,
        "Heart_Rate_BPM": rng.normal(74, 3, days).round(1),
        "Oxygen_Sat_Percentage": np.clip(rng.normal(98.0, 0.4, days), 85, 100).round(1),
        "Systolic_BP_mmHg": rng.normal(122, 4, days).round(1),
        "Voice_Instability_Index": np.clip(rng.normal(12.2, 1.5, days), 0, 100).round(2),
        "Fundamental_Pitch_Hz": rng.normal(135.0, 5.0, days).round(1),
    }).set_index("Date")


ML_FEATURES = [
    "Heart_Rate_BPM",
    "Oxygen_Sat_Percentage",
    "Systolic_BP_mmHg",
    "Voice_Instability_Index",
    "Fundamental_Pitch_Hz",
]
MIN_ROWS_FOR_ML = 5


def detect_anomalies(history: pd.DataFrame, contamination: float = 0.05):
    data = history[ML_FEATURES].dropna()
    if len(data) < MIN_ROWS_FOR_ML:
        return None, f"Need at least {MIN_ROWS_FOR_ML} entries for anomaly detection."

    train = data.iloc[:-1]
    scaler = StandardScaler().fit(train)
    model = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=42,
    ).fit(scaler.transform(train))

    scaled_all = scaler.transform(data)
    scores = model.decision_function(scaled_all)
    flags = model.predict(scaled_all) == -1

    result = pd.DataFrame(
        {"Anomaly_Score": scores.round(3), "Is_Anomaly": flags},
        index=data.index,
    )
    return result, None


def forecast_trend(series: pd.Series, horizon: int = 7, window: int = 30):
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
    direction = "Stable" if rel < 0.05 else ("Rising" if slope > 0 else "Falling")

    return {
        "slope": slope,
        "direction": direction,
        "r2": float(model
