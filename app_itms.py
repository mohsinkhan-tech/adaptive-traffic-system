# app_itms.py
# AI-Enabled Intelligent Traffic Management System (ITMS)
# Streamlit single-file prototype that runs locally and on low-cost edge devices

import os
import time
import threading
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
import streamlit as st

# Optional libraries (loaded lazily)
_TF = None
_TORCH = None
_ULTRALYTICS = None
_PAHO = None
_SKLEARN = None


# ----------------------------
# Utility: Lazy imports
# ----------------------------
def _lazy_imports():
    global _TF, _TORCH, _ULTRALYTICS, _PAHO, _SKLEARN
    try:
        import tensorflow as tf
        _TF = tf
    except Exception:
        _TF = None
    try:
        import torch
        _TORCH = torch
    except Exception:
        _TORCH = None
    try:
        from ultralytics import YOLO
        _ULTRALYTICS = YOLO
    except Exception:
        _ULTRALYTICS = None
    try:
        import paho.mqtt.client as mqtt
        _PAHO = mqtt
    except Exception:
        _PAHO = None
    try:
        import sklearn
        _SKLEARN = sklearn
    except Exception:
        _SKLEARN = None


# ----------------------------
# Privacy Anonymization
# ----------------------------
class Anonymizer:
    def __init__(self):
        cascade_path = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_path, 'haarcascade_frontalface_default.xml'))
        self.plate_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_path, 'haarcascade_russian_plate_number.xml'))

    def _blur_regions(self, frame, regions):
        for (x, y, w, h) in regions:
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            roi = cv2.GaussianBlur(roi, (31, 31), 0)
            frame[y:y+h, x:x+w] = roi
        return frame

    def anonymize(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.2, 4)
        plates = self.plate_cascade.detectMultiScale(gray, 1.1, 3)
        frame = self._blur_regions(frame, faces)
        frame = self._blur_regions(frame, plates)
        return frame


# ----------------------------
# Vehicle Detection & Counting
# ----------------------------
@dataclass
class DetectionResult:
    boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    count: int = 0


class VehicleDetector:
    def __init__(self, use_yolo: bool = True, conf_thres: float = 0.4,
                 target_classes: Optional[List[str]] = None):
        _lazy_imports()
        self.use_yolo = use_yolo and (_ULTRALYTICS is not None)
        self.conf_thres = conf_thres
        self.target_classes = target_classes or ['car', 'truck', 'bus', 'motorbike']

        self.yolo_model = None
        if self.use_yolo:
            try:
                self.yolo_model = _ULTRALYTICS('yolov8n.pt')
            except Exception:
                self.use_yolo = False

        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=25, detectShadows=True)

    def _detect_yolo(self, frame) -> DetectionResult:
        res = DetectionResult()
        if self.yolo_model is None:
            return res
        results = self.yolo_model.predict(frame, conf=self.conf_thres, verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls_id = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                xyxy = b.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1
                label = r.names.get(cls_id, str(cls_id)).lower()
                if label in self.target_classes:
                    res.boxes.append((x1, y1, w, h))
                    res.classes.append(label)
                    res.confidences.append(conf)
        res.count = len(res.boxes)
        return res

    def _detect_bg(self, frame) -> DetectionResult:
        res = DetectionResult()
        fg = self.bg.apply(frame)
        fg = cv2.medianBlur(fg, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=2)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 800:
                continue
            x, y, w, h = cv2.boundingRect(c)
            res.boxes.append((x, y, w, h))
            res.classes.append('vehicle')
            res.confidences.append(0.5)
        res.count = len(res.boxes)
        return res

    def detect(self, frame) -> DetectionResult:
        if self.use_yolo:
            return self._detect_yolo(frame)
        else:
            return self._detect_bg(frame)


# ----------------------------
# Forecasting (LSTM or Fallback)
# ----------------------------
class FlowForecaster:
    def __init__(self, lookback: int = 12, horizon: int = 6):
        _lazy_imports()
        self.lookback = lookback
        self.horizon = horizon
        self.model = None
        self.mode = None  # 'lstm', 'rf', or 'naive'

    def _build_lstm(self, input_shape):
        tf = _TF
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.horizon)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def fit(self, series: List[float], epochs: int = 10):
        series = np.array(series, dtype=np.float32)
        if len(series) < self.lookback + self.horizon + 10:
            self.mode = 'naive'
            return

        X, y = [], []
        for i in range(len(series) - self.lookback - self.horizon + 1):
            X.append(series[i:i + self.lookback])
            y.append(series[i + self.lookback:i + self.lookback + self.horizon])
        X = np.array(X)
        y = np.array(y)
        X = X.reshape((-1, self.lookback, 1))

        if _TF is not None:
            try:
                self.model = self._build_lstm((self.lookback, 1))
                self.model.fit(X, y, epochs=epochs, batch_size=16, verbose=0)
                self.mode = 'lstm'
                return
            except Exception:
                self.model = None

        if _SKLEARN is not None:
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=100)
            self.model.fit(X.reshape((X.shape[0], X.shape[1])), y.mean(axis=1))
            self.mode = 'rf'
        else:
            self.mode = 'naive'

    def predict(self, recent_series: List[float]) -> List[float]:
        s = np.array(recent_series[-self.lookback:], dtype=np.float32)
        if len(s) < self.lookback:
            pad = self.lookback - len(s)
            s = np.concatenate([np.zeros(pad, dtype=np.float32), s])

        if self.mode == 'lstm' and self.model is not None:
            pred = self.model.predict(s.reshape((1, self.lookback, 1)), verbose=0)[0]
            return pred.tolist()
        elif self.mode == 'rf' and self.model is not None:
            yhat = self.model.predict(s.reshape(1, -1))[0]
            return [float(max(0.0, yhat))] * self.horizon
        else:
            last = float(s[-1]) if len(s) else 0.0
            return [last] * self.horizon


# ----------------------------
# RL Controller (DQN or Heuristic)
# ----------------------------
@dataclass
class TLState:
    q_ns: int
    q_ew: int
    phase: int  # 0: NS green, 1: EW green


class SimpleIntersectionSim:
    def __init__(self, sat_flow: int = 4, arrival_rate_ns: float = 2.0,
                 arrival_rate_ew: float = 2.0):
        self.sat_flow = sat_flow
        self.arrival_rate_ns = arrival_rate_ns
        self.arrival_rate_ew = arrival_rate_ew
        self.reset()

    def reset(self) -> TLState:
        self.q_ns = 10
        self.q_ew = 10
        self.phase = 0
        return TLState(self.q_ns, self.q_ew, self.phase)

    def step(self, action: int) -> Tuple[TLState, float]:
        if action == 1:
            self.phase = 1 - self.phase
            penalty = 1.0
        else:
            penalty = 0.0

        arr_ns = np.random.poisson(self.arrival_rate_ns)
        arr_ew = np.random.poisson(self.arrival_rate_ew)
        self.q_ns += arr_ns
        self.q_ew += arr_ew

        if self.phase == 0:
            dep_ns = min(self.q_ns, self.sat_flow)
            dep_ew = min(self.q_ew, 1)
        else:
            dep_ew = min(self.q_ew, self.sat_flow)
            dep_ns = min(self.q_ns, 1)

        self.q_ns -= dep_ns
        self.q_ew -= dep_ew

        reward = -(self.q_ns + self.q_ew) - penalty
        return TLState(self.q_ns, self.q_ew, self.phase), reward


class DQNAgent:
    def __init__(self, state_dim=3, action_dim=2, gamma=0.95, lr=1e-3):
        _lazy_imports()
        self.use_torch = _TORCH is not None
        self.gamma = gamma
        self.action_dim = action_dim
        self.memory = []
        self.max_mem = 5000
        self.batch = 64
        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.995

        if self.use_torch:
            import torch.nn as nn
            self.device = _TORCH.device('cuda' if _TORCH.cuda.is_available() else 'cpu')

            class Net(nn.Module):
                def __init__(self, state_dim, action_dim):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(state_dim, 64), nn.ReLU(),
                        nn.Linear(64, 64), nn.ReLU(),
                        nn.Linear(64, action_dim)
                    )

                def forward(self, x):
                    return self.net(x)

            self.q = Net(state_dim, action_dim).to(self.device)
            self.opt = _TORCH.optim.Adam(self.q.parameters(), lr=lr)
            self.loss_fn = _TORCH.nn.MSELoss()
        else:
            self.device = None

    def select_action(self, state: TLState) -> int:
        if self.use_torch:
            if np.random.rand() < self.eps:
                return np.random.randint(self.action_dim)
            s = _TORCH.tensor(
                [state.q_ns, state.q_ew, state.phase],
                dtype=_TORCH.float32).to(self.device)
            with _TORCH.no_grad():
                qvals = self.q(s)
            return int(_TORCH.argmax(qvals).item())
        else:
            longer_is_ns = state.q_ns > state.q_ew * 1.2
            longer_is_ew = state.q_ew > state.q_ns * 1.2
            want_phase = 0 if longer_is_ns else (1 if longer_is_ew else state.phase)
            return 0 if want_phase == state.phase else 1

    def remember(self, s, a, r, s2, done=False):
        if not self.use_torch:
            return
        self.memory.append((s, a, r, s2, done))
        if len(self.memory) > self.max_mem:
            self.memory = self.memory[-self.max_mem:]

    def learn(self):
        if not self.use_torch:
            return
        if len(self.memory) < self.batch:
            return
        batch = np.random.choice(len(self.memory), self.batch, replace=False)
        s_list, a_list, r_list, s2_list, d_list = [], [], [], [], []
        for idx in batch:
            s, a, r, s2, d = self.memory[idx]
            s_list.append(s)
            a_list.append(a)
            r_list.append(r)
            s2_list.append(s2)
            d_list.append(d)

        s  = _TORCH.tensor(s_list,  dtype=_TORCH.float32).to(self.device)
        a  = _TORCH.tensor(a_list,  dtype=_TORCH.long).to(self.device)
        r  = _TORCH.tensor(r_list,  dtype=_TORCH.float32).to(self.device)
        s2 = _TORCH.tensor(s2_list, dtype=_TORCH.float32).to(self.device)
        d  = _TORCH.tensor(d_list,  dtype=_TORCH.float32).to(self.device)

        qvals = self.q(s).gather(1, a.view(-1, 1)).squeeze()
        with _TORCH.no_grad():
            qnext = self.q(s2).max(1)[0]
        target = r + self.gamma * qnext * (1.0 - d)
        loss = self.loss_fn(qvals, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.eps = max(self.eps_min, self.eps * self.eps_decay)


# ----------------------------
# MQTT Hooks (optional)
# ----------------------------
class MQTTHook:
    def __init__(self, host='localhost', port=1883, topic='itms/metrics'):
        _lazy_imports()
        self.client = None
        self.topic = topic
        if _PAHO is not None:
            try:
                self.client = _PAHO.Client()
                self.client.connect(host, port, 60)
                self.client.loop_start()
            except Exception:
                self.client = None

    def publish(self, payload: Dict):
        if self.client is None:
            return
        try:
            self.client.publish(self.topic, json.dumps(payload))
        except Exception:
            pass


# ----------------------------
# Stream Processor Thread
# ----------------------------
class VideoProcessor:
    def __init__(self, src=0, width=640, height=360, use_yolo=True, anonymize=True):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.detector = VehicleDetector(use_yolo=use_yolo)
        self.anon = Anonymizer() if anonymize else None
        self.running = False
        self.frame = None
        self.count = 0
        self.lock = threading.Lock()
        self.fps = 0.0
        self.last_count_time = 0.0

    def start(self):
        self.running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self):
        self.running = False
        try:
            self.cap.release()
        except Exception:
            pass

    def _loop(self):
        prev = time.time()
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            frame = cv2.resize(frame, (640, 360))
            det = self.detector.detect(frame)
            for (x, y, w, h), cls, conf in zip(det.boxes, det.classes, det.confidences):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{cls}:{conf:.2f}", (x, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if self.anon is not None:
                frame = self.anon.anonymize(frame)
            with self.lock:
                self.frame = frame
                self.count = det.count
            now = time.time()
            dt = now - prev
            self.fps = 1.0 / dt if dt > 0 else 0.0
            prev = now
        try:
            self.cap.release()
        except Exception:
            pass


# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="AI ITMS", layout="wide")
st.title("🚦 AI-Enabled Intelligent Traffic Management System (ITMS)")

# --- Session state init ---
if 'vp' not in st.session_state:
    st.session_state.vp = None
if 'counts' not in st.session_state:
    st.session_state.counts = []
if 'forecaster' not in st.session_state:
    st.session_state.forecaster = FlowForecaster(lookback=12, horizon=6)
if 'forecaster_trained' not in st.session_state:
    st.session_state.forecaster_trained = False
if 'sim' not in st.session_state:
    st.session_state.sim = SimpleIntersectionSim()
if 'agent' not in st.session_state:
    st.session_state.agent = DQNAgent()
if 'mqtt' not in st.session_state:
    st.session_state.mqtt = MQTTHook()
if 'last_count_time' not in st.session_state:
    st.session_state.last_count_time = 0.0

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    src_type = st.selectbox("Video Source", ["Webcam (0)", "File"], index=0)
    video_src = 0
    if src_type == "File":
        uploaded = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov"])
        if uploaded is not None:
            tmp_path = os.path.join(".", "_tmp_video.mp4")
            with open(tmp_path, 'wb') as f:
                f.write(uploaded.read())
            video_src = tmp_path
    use_yolo = st.checkbox("Use YOLOv8 (if available)", value=True)
    anonymize = st.checkbox("Anonymize faces/plates", value=True)
    st.write("Optional integrations:")
    st.caption("MQTT publisher is auto-enabled if paho-mqtt is installed.")

    if st.button("Start Processing", type="primary"):
        if st.session_state.vp is not None:
            st.session_state.vp.stop()
        st.session_state.vp = VideoProcessor(
            src=video_src, use_yolo=use_yolo, anonymize=anonymize)
        st.session_state.vp.start()

    if st.button("Stop Processing"):
        if st.session_state.vp is not None:
            st.session_state.vp.stop()
            st.session_state.vp = None

# --- Layout ---
col_video, col_metrics = st.columns([2, 1])
with col_video:
    st.subheader("Live Feed & Detection")
    frame_placeholder = st.empty()
with col_metrics:
    st.subheader("Metrics")
    # FIX: use consistent placeholder names
    count_placeholder = st.empty()
    fps_placeholder = st.empty()

col_forecast, col_control = st.columns(2)
with col_forecast:
    st.subheader("Traffic Flow Forecasting")
    st.caption("Model: LSTM if TensorFlow available, else RandomForest/Naive fallback.")
    retrain = st.button("(Re)train Forecaster on recent counts")
    forecast_out = st.empty()
with col_control:
    st.subheader("Signal Control (RL)")
    st.caption("Agent: DQN if PyTorch available, else heuristic controller.")
    control_action = st.empty()

# --- Simulator panel ---
with st.expander("🚦 Simple Intersection Simulator (for offline testing)"):
    steps = st.slider("Steps", 10, 200, 50, 10)
    if st.button("Run Simulation"):
        sim = st.session_state.sim
        agent = st.session_state.agent
        s = sim.reset()
        total_reward = 0.0
        for t in range(steps):
            a = agent.select_action(s)
            s2, r = sim.step(a)
            if agent.use_torch:
                s_vec = [s.q_ns, s.q_ew, s.phase]
                s2_vec = [s2.q_ns, s2.q_ew, s2.phase]
                agent.remember(s_vec, a, r, s2_vec, False)
                agent.learn()
            total_reward += r
            s = s2
        st.info(
            f"Simulation finished. Total reward: {total_reward:.2f}. "
            f"Final queues NS={s.q_ns}, EW={s.q_ew}, phase={s.phase}"
        )

# --- Handle retrain button (FIX: use session state flag, not button bool) ---
if retrain:
    counts = st.session_state.counts
    forecaster = st.session_state.forecaster
    if len(counts) > forecaster.lookback + 10:
        forecaster.fit(counts, epochs=5)
        st.session_state.forecaster_trained = True
        st.success("Forecaster retrained.")
    else:
        st.warning("Not enough data yet to train the forecaster.")

# --- Live frame display (single-pass, no while loop) ---
vp = st.session_state.vp

if vp is None:
    st.info("Press **Start Processing** in the sidebar to begin.")
else:
    with vp.lock:
        frame = None if vp.frame is None else vp.frame.copy()
        count = vp.count
        fps = vp.fps

    if frame is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels='RGB', use_container_width=True)

    # FIX: use the correctly named placeholders
    count_placeholder.metric("Vehicles (last frame)", count if frame is not None else 0)
    fps_placeholder.metric("Processing FPS", round(fps, 2) if frame is not None else 0.0)

    # Append count to time-series roughly every second
    now = time.time()
    if now - st.session_state.last_count_time > 1.0 and frame is not None:
        st.session_state.counts.append(count)
        if len(st.session_state.counts) > 600:
            st.session_state.counts = st.session_state.counts[-600:]
        st.session_state.last_count_time = now
        st.session_state.mqtt.publish({
            'ts': int(now),
            'count': int(count),
            'fps': float(fps)
        })

    # Forecast display
    preds = st.session_state.forecaster.predict(st.session_state.counts)
    forecast_out.write({
        "horizon": len(preds),
        "predicted_count_per_step": [round(float(p), 2) for p in preds]
    })

    # RL control decision
    def _recent_avg(arr, k=5):
        if not arr:
            return 0
        return int(np.mean(arr[-k:]))

    counts = st.session_state.counts
    avg_ns = _recent_avg(counts[-10::2])
    avg_ew = _recent_avg(counts[-9::2])
    dummy_state = TLState(q_ns=avg_ns, q_ew=avg_ew, phase=0)
    a = st.session_state.agent.select_action(dummy_state)
    control_action.info(
        f"Recommended action: {'Switch phase' if a == 1 else 'Keep phase'} "
        f"| State approx NS={avg_ns}, EW={avg_ew}"
    )

    # FIX: use st.rerun() instead of a blocking while loop
    time.sleep(0.05)
    st.rerun()