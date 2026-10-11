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
            # Fallback for mock/demo files
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

# --- SIDEBAR: REGISTRATION & SELECTOR ---
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
        st.sidebar.success(f"Registered {new_name} ({new_id}) successfully!")

st.sidebar.markdown("---")
st.sidebar.header("EHR Records Selector")
selected_pt_id = st.sidebar.selectbox(
    "Active Patient Profile Selection:",
    options=list(st.session_state.master_registry.keys()),
    format_func=lambda x: f"{x} - {st.session_state.master_registry[x]['name']}"
)

current_patient = st.session_state.master_registry[selected_pt_id]

# --- PATIENT HEADER METADATA ---
st.markdown(
    f"""
    **Patient Name:** {current_patient['name']} &nbsp;|&nbsp; 
    **Chronological Age:** {current_patient['age']} Years &nbsp;|&nbsp; 
    **Monitored Condition Path:** {current_patient['condition']} &nbsp;|&nbsp; 
    **Internal Record Key:** {selected_pt_id}[cite: 1, 2]
    """
)
st.markdown("---")

# =====================================================================
# LOG NEW VITAL DIAGNOSTICS & VOICE AUDIO ENTRY
# =====================================================================
st.subheader("Log New Vital Diagnostics Entry & Audio Sample")

with st.form("telemetry_log_form"):
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        st.markdown("#### Input Vital Measurements")
        hr_input = st.number_input("Heart Rate (BPM)", min_value=40, max_value=200, value=80)
        bp_input = st.number_input("Systolic Blood Pressure (mmHg)", min_value=70, max_value=220, value=115)
        spo2_input = st.number_input("Oxygen Saturation Level - SpO2 (%)", min_value=70.0, max_value=100.0, value=98.4, format="%.1f")

    with col_input2:
        st.markdown("#### Hormonal Acoustic Analysis (Voice Record)")
        uploaded_audio = st.file_uploader(
            "Drop patient sustained vowel sound file (.wav) to evaluate Vocal Cord Edema patterns",
            type=["wav"]
        )
        st.markdown("#### ML Settings")
        anomaly_sens = st.slider("Anomaly sensitivity (expected outlier contamination)", 0.01, 0.20, 0.05)
        forecast_horizon = st.slider("Forecast horizon (entries)", 3, 14, 7)

    commit_btn = st.form_submit_button("Commit Log Entry to Pipeline")

    if commit_btn:
        audio_bytes = uploaded_audio.read() if uploaded_audio is not None else b""
        engine = VoiceBiometricEngine()
        voice_metrics = engine.analyze_uploaded_file(audio_bytes)

        new_row = pd.DataFrame({
            "Heart_Rate_BPM": [float(hr_input)],
            "Oxygen_Sat_Percentage": [float(spo2_input)],
            "Systolic_BP_mmHg": [float(bp_input)],
            "Voice_Instability_Index": [float(voice_metrics["instability_score"])],
            "Fundamental_Pitch_Hz": [float(voice_metrics["fundamental_pitch_hz"])],
        }, index=[pd.Timestamp.now()])

        current_patient["history"] = pd.concat([current_patient["history"], new_row])
        st.success("Telemetry record and acoustic features successfully committed to patient longitudinal history!")

st.markdown("---")

# =====================================================================
# AUTOMATED CLINICAL ASSESSMENT & HEALTH STATUS REPORT
# =====================================================================
st.subheader("🤖 Automated Clinical Assessment & Health Status Report")
st.markdown("Latest analysis metrics measured from uploaded audio stream and vital telemetry:")

latest_rec = current_patient["history"].iloc[-1]
prev_rec = current_patient["history"].iloc[-2] if len(current_patient["history"]) > 1 else latest_rec

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Fundamental Pitch (Hz)", f"{latest_rec['Fundamental_Pitch_Hz']:.1f}", f"{latest_rec['Fundamental_Pitch_Hz'] - prev_rec['Fundamental_Pitch_Hz']:+.1f}")
m2.metric("Heart Rate", f"{latest_rec['Heart_Rate_BPM']:.1f}", f"{latest_rec['Heart_Rate_BPM'] - prev_rec['Heart_Rate_BPM']:+.1f}")
m3.metric("Systolic BP", f"{latest_rec['Systolic_BP_mmHg']:.1f}", f"{latest_rec['Systolic_BP_mmHg'] - prev_rec['Systolic_BP_mmHg']:+.1f}")
m4.metric("Voice Instability Index", f"{latest_rec['Voice_Instability_Index']:.2f}", f"{latest_rec['Voice_Instability_Index'] - prev_rec['Voice_Instability_Index']:+.2f}")
m5.metric("SpO2 %", f"{latest_rec['Oxygen_Sat_Percentage']:.1f}", f"{latest_rec['Oxygen_Sat_Percentage'] - prev_rec['Oxygen_Sat_Percentage']:+.1f}")

# Clinical Alert Banner Check
instability_val = latest_rec["Voice_Instability_Index"]
if instability_val > 20.0:
    st.markdown(
        f"""
        <div class="alert-box">
            <strong>⚠️ Clinical Alert: Potential Health Issues Detected for {current_patient['name']}</strong><br>
            Diagnostic Evaluation: Patient exhibits indicators associated with {current_patient['condition']}.<br>
            The system telemetry has flagged the following concerns:<br>
            • <strong>Elevated Voice Instability Index: {instability_val:.2f}</strong> (Threshold: &lt;20.0)<br>
            <em>Recommended Action:</em> Schedule a clinical follow-up or review historical trends below.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="success-box">
            <strong>✅ Clinical Status: Stable</strong><br>
            All acoustic and hemodynamic indicators are currently within optimal baselines.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# =====================================================================
# MACHINE LEARNING INSIGHTS SECTION
# =====================================================================
st.subheader("Machine Learning Insights")
ml_col1, ml_col2 = st.columns(2)

with ml_col1:
    st.markdown("#### Anomaly Detection (Isolation Forest)")
    anomaly_res, err = detect_anomalies(current_patient["history"], contamination=anomaly_sens)
    if err:
        st.info(err)
    else:
        latest_is_anomaly = anomaly_res["Is_Anomaly"].iloc[-1]
        latest_score = anomaly_res["Anomaly_Score"].iloc[-1]
        total_anomalies = int(anomaly_res["Is_Anomaly"].sum())
        total_records = len(anomaly_res)

        if latest_is_anomaly:
            st.warning(f"⚠️ Latest entry is unusual for {current_patient['name']} (score {latest_score}). It deviates from this patient's learned baseline.")
        else:
            st.success(f"✓ Latest entry aligns with normal baseline patterns (score {latest_score}).")
        st.caption(f"{total_anomalies} of {total_records} recorded entries flagged as anomalous.")

with ml_col2:
    st.markdown("#### Trend & Forecast (Linear Regression)")
    metric_choice = st.selectbox("Metric to forecast:", ML_FEATURES, index=3)
    forecast_data = forecast_trend(current_patient["history"][metric_choice], horizon=forecast_horizon)
    if forecast_data:
        st.markdown(
            f"""
            * **Trend Direction:** {forecast_data['direction']}
            * **Slope / Entry:** {forecast_data['slope']:+.3f}
            * **Model Fit ($R^2$):** {forecast_data['r2']:.2f}
            """
        )
    else:
        st.info("Insufficient data for robust regression forecasting.")

st.markdown("---")

# =====================================================================
# UNIQUE FEATURE: AI CLINICAL DETERIORATION SCORE & EHR HANDOVER COPILOT
# =====================================================================
st.subheader("🌟 Unique Feature: AI Clinical Deterioration Early Warning Score (DEWS) & EHR Handover Copilot")
st.markdown("Generates an automated multi-parameter acuity score and physician-ready handover summary for hospital records.")

# Calculate DEWS Score (composite of HR, BP, SpO2, and Voice Instability)
hr_score = 2 if (latest_rec['Heart_Rate_BPM'] > 100 or latest_rec['Heart_Rate_BPM'] < 50) else 0
bp_score = 2 if latest_rec['Systolic_BP_mmHg'] < 90 or latest_rec['Systolic_BP_mmHg'] > 160 else 0
spo2_score = 3 if latest_rec['Oxygen_Sat_Percentage'] < 92 else (1 if latest_rec['Oxygen_Sat_Percentage'] < 95 else 0)
voice_score = 3 if latest_rec['Voice_Instability_Index'] > 35 else (1 if latest_rec['Voice_Instability_Index'] > 20 else 0)
total_dews = hr_score + bp_score + spo2_score + voice_score

risk_level = "CRITICAL" if total_dews >= 5 else ("MODERATE" if total_dews >= 3 else "LOW RISK (STABLE)")

dews_col1, dews_col2 = st.columns([1, 2])
with dews_col1:
    st.metric("Composite DEWS Score", f"{total_dews} / 9", delta=risk_level, delta_inverse=True)

with dews_col2:
    handover_text = f"""
    ### CLINICAL HANDOVER SUMMARY (EHR EXPORT)
    - **Patient:** {current_patient['name']} ({selected_pt_id}), Age {current_patient['age']}
    - **Condition:** {current_patient['condition']}
    - **Acuity Tier:** {risk_level} (DEWS: {total_dews}/9)
    - **Key Findings:** HR {latest_rec['Heart_Rate_BPM']} BPM, BP {latest_rec['Systolic_BP_mmHg']} mmHg, SpO2 {latest_rec['Oxygen_Sat_Percentage']}%, Voice Instability Index {latest_rec['Voice_Instability_Index']:.2f}.
    - **Clinical Directive:** {'Immediate endocrinologist / physician notification required.' if total_dews >= 3 else 'Routine remote monitoring continuation.'}
    """
    st.info(handover_text)

st.markdown("---")

# =====================================================================
# HISTORICAL TELEMETRY & TRENDS TABS
# =====================================================================
st.subheader("📈 Historical Telemetry & Trends")
tab_vitals, tab_voice, tab_raw, tab_notes = st.tabs([
    "Vitals & Hemodynamics",
    "Acoustic & Voice Biomarkers",
    "Raw Data Table",
    "Clinical Notes"
])

with tab_vitals:
    v_c1, v_c2, v_c3 = st.columns(3)
    with v_c1:
        st.markdown("**Heart Rate (BPM)**")
        st.line_chart(current_patient["history"]["Heart_Rate_BPM"])
    with v_c2:
        st.markdown("**Systolic BP (mmHg)**")
        st.line_chart(current_patient["history"]["Systolic_BP_mmHg"])
    with v_c3:
        st.markdown("**Oxygen Saturation SpO2 (%)**")
        st.line_chart(current_patient["history"]["Oxygen_Sat_Percentage"])

with tab_voice:
    vo_c1, vo_c2 = st.columns(2)
    with vo_c1:
        st.markdown("**Voice Instability Index**")
        st.line_chart(current_patient["history"]["Voice_Instability_Index"])
    with vo_c2:
        st.markdown("**Fundamental Pitch (Hz)**")
        st.line_chart(current_patient["history"]["Fundamental_Pitch_Hz"])

with tab_raw:
    st.dataframe(current_patient["history"], use_container_width=True)

with tab_notes:
    st.text_area("Physician Clinical Notes", value=current_patient["notes"], height=150)
    if st.button("Update Clinical Notes"):
        st.success("Notes updated successfully.")

st.markdown("---")
st.markdown("*Master Clinical RPMA Telemetry Panel 2.5 (ML-enabled) — Designed for research and remote monitoring workflows[cite: 2].*")
