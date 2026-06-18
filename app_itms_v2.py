# app_itms_v2.py
# AI-Powered Adaptive Traffic Signal Control System
# Authors: Asad Irfan (65117), Mohsin Khan (62876)
# UPDATED: Added ML (Random Forest), ANN, Kaggle dataset support,
#          accuracy comparison ML alone vs ML + ANN, removed video dependency

import os
import time
import json
import threading
import warnings
warnings.filterwarnings('ignore')

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st

# Optional libraries (loaded lazily)
_TF = None
_TORCH = None
_SKLEARN = None
_PAHO = None

# ----------------------------
# Utility: Lazy imports
# ----------------------------
def _lazy_imports():
    global _TF, _TORCH, _SKLEARN, _PAHO
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
        import sklearn
        _SKLEARN = sklearn
    except Exception:
        _SKLEARN = None
    try:
        import paho.mqtt.client as mqtt
        _PAHO = mqtt
    except Exception:
        _PAHO = None

_lazy_imports()

# ============================================================
# SECTION 1 — DATASET LOADING (Kaggle CSV or Synthetic)
# ============================================================

KAGGLE_COLUMNS = {
    # Maps common Kaggle traffic dataset column names → our internal names
    'vehicle_count':      ['vehicle_count', 'count', 'vehicles', 'total_vehicles', 'Volume'],
    'hour':               ['hour', 'Hour', 'time_hour', 'hr'],
    'day_of_week':        ['day_of_week', 'day', 'Day', 'weekday'],
    'speed':              ['speed', 'Speed', 'avg_speed', 'average_speed'],
    'congestion_level':   ['congestion_level', 'congestion', 'Congestion', 'traffic_level'],
    'queue_ns':           ['queue_ns', 'ns_queue', 'north_south_queue'],
    'queue_ew':           ['queue_ew', 'ew_queue', 'east_west_queue'],
    'signal_phase':       ['signal_phase', 'phase', 'Phase', 'signal'],
    'waiting_time':       ['waiting_time', 'wait_time', 'delay', 'Delay'],
}

def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def load_kaggle_csv(uploaded_file) -> pd.DataFrame:
    """Load and normalize a Kaggle traffic CSV to standard column names."""
    df = pd.read_csv(uploaded_file)
    rename_map = {}
    for internal_name, candidates in KAGGLE_COLUMNS.items():
        found = _find_column(df, candidates)
        if found and found != internal_name:
            rename_map[found] = internal_name
    df = df.rename(columns=rename_map)

    # Ensure essential columns exist; synthesize if missing
    if 'vehicle_count' not in df.columns:
        df['vehicle_count'] = np.random.poisson(20, len(df))
    if 'hour' not in df.columns:
        df['hour'] = np.arange(len(df)) % 24
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = (np.arange(len(df)) // 24) % 7
    if 'speed' not in df.columns:
        df['speed'] = np.clip(60 - df['vehicle_count'] * 1.2, 5, 80)
    if 'congestion_level' not in df.columns:
        df['congestion_level'] = pd.cut(
            df['vehicle_count'],
            bins=[0, 10, 20, 35, np.inf],
            labels=[0, 1, 2, 3]
        ).astype(int)
    if 'queue_ns' not in df.columns:
        df['queue_ns'] = (df['vehicle_count'] * np.random.uniform(0.4, 0.6, len(df))).astype(int)
    if 'queue_ew' not in df.columns:
        df['queue_ew'] = (df['vehicle_count'] - df['queue_ns']).clip(0).astype(int)
    if 'waiting_time' not in df.columns:
        df['waiting_time'] = df['queue_ns'] * 3 + df['queue_ew'] * 3
    if 'signal_phase' not in df.columns:
        df['signal_phase'] = (df['queue_ns'] < df['queue_ew']).astype(int)

    df = df.dropna().reset_index(drop=True)
    return df


def generate_synthetic_dataset(n: int = 2000) -> pd.DataFrame:
    """Generate a realistic synthetic traffic dataset for demo/testing."""
    np.random.seed(42)
    hours = np.tile(np.arange(24), n // 24 + 1)[:n]
    days  = np.repeat(np.arange(7), n // 7 + 1)[:n]

    # Rush hour pattern
    base_flow = 10 + 25 * np.exp(-0.5 * ((hours - 8) / 1.5) ** 2) \
                   + 20 * np.exp(-0.5 * ((hours - 17) / 1.5) ** 2)
    vehicle_count = np.clip(
        base_flow + np.random.normal(0, 4, n), 1, 60
    ).astype(int)

    speed         = np.clip(70 - vehicle_count * 0.9 + np.random.normal(0, 3, n), 5, 80)
    congestion    = pd.cut(vehicle_count, bins=[0,10,20,35,999], labels=[0,1,2,3]).astype(int)
    queue_ns      = (vehicle_count * np.random.uniform(0.35, 0.65, n)).astype(int)
    queue_ew      = (vehicle_count - queue_ns).clip(0)
    waiting_time  = queue_ns * 3 + queue_ew * 3 + np.random.randint(0, 10, n)
    signal_phase  = (queue_ns < queue_ew).astype(int)

    return pd.DataFrame({
        'hour': hours, 'day_of_week': days,
        'vehicle_count': vehicle_count, 'speed': speed.round(1),
        'congestion_level': congestion,
        'queue_ns': queue_ns, 'queue_ew': queue_ew,
        'waiting_time': waiting_time, 'signal_phase': signal_phase,
    })


# ============================================================
# SECTION 2 — FEATURE ENGINEERING
# ============================================================

def build_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build feature matrix X and target y for signal phase prediction.
    Returns (X, y, feature_names)
    """
    feature_cols = ['hour', 'day_of_week', 'vehicle_count',
                    'speed', 'queue_ns', 'queue_ew', 'waiting_time']
    # Add congestion if present
    if 'congestion_level' in df.columns:
        feature_cols.append('congestion_level')

    # Keep only columns that actually exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].values.astype(np.float32)
    y = df['signal_phase'].values.astype(int)
    return X, y, feature_cols


# ============================================================
# SECTION 3 — ML MODEL (Random Forest)
# ============================================================

class MLPredictor:
    """Random Forest classifier for signal phase prediction."""

    def __init__(self, n_estimators: int = 100):
        self.n_estimators = n_estimators
        self.model = None
        self.trained = False
        self.feature_importances_ = None

    def train(self, X: np.ndarray, y: np.ndarray):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X, y)
        self.trained = True
        self.feature_importances_ = self.model.feature_importances_

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trained:
            return np.zeros(len(X), dtype=int)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.trained:
            return np.ones((len(X), 2)) * 0.5
        return self.model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        from sklearn.metrics import (accuracy_score, precision_score,
                                     recall_score, f1_score, confusion_matrix)
        if not self.trained:
            return {}
        y_pred = self.predict(X)
        return {
            'accuracy':  round(accuracy_score(y, y_pred) * 100, 2),
            'precision': round(precision_score(y, y_pred, zero_division=0) * 100, 2),
            'recall':    round(recall_score(y, y_pred, zero_division=0) * 100, 2),
            'f1':        round(f1_score(y, y_pred, zero_division=0) * 100, 2),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
        }


# ============================================================
# SECTION 4 — ANN MODEL (Artificial Neural Network)
# ============================================================

class ANNPredictor:
    """
    Artificial Neural Network for signal phase prediction.
    Uses TensorFlow/Keras if available, else falls back to a
    simple NumPy-based 2-layer perceptron.
    """

    def __init__(self, hidden_units: List[int] = [64, 32], epochs: int = 30):
        _lazy_imports()
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.model = None
        self.trained = False
        self.use_tf = _TF is not None
        self.history = None
        self._w = []   # weights for numpy fallback
        self._b = []   # biases for numpy fallback
        self._input_dim = None

    # --- TensorFlow path ---
    def _build_tf_model(self, input_dim: int):
        tf = _TF
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(self.hidden_units[0], activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(self.hidden_units[1], activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    # --- NumPy fallback (simple MLP with sigmoid activations) ---
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _relu(self, x):
        return np.maximum(0, x)

    def _init_numpy_weights(self, input_dim: int):
        """He initialization."""
        dims = [input_dim] + self.hidden_units + [1]
        self._w = []
        self._b = []
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])
            self._w.append(np.random.randn(dims[i], dims[i+1]) * scale)
            self._b.append(np.zeros(dims[i+1]))

    def _forward_numpy(self, X: np.ndarray) -> np.ndarray:
        a = X
        for i, (w, b) in enumerate(zip(self._w, self._b)):
            z = a @ w + b
            if i < len(self._w) - 1:
                a = self._relu(z)
            else:
                a = self._sigmoid(z)
        return a.squeeze()

    def _train_numpy(self, X: np.ndarray, y: np.ndarray):
        """Mini-batch SGD with binary cross-entropy."""
        lr = 0.01
        batch_size = 64
        n = len(X)
        self._init_numpy_weights(X.shape[1])
        for epoch in range(self.epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, batch_size):
                batch = idx[start:start+batch_size]
                Xb, yb = X[batch], y[batch].astype(np.float32)
                # Forward
                activations = [Xb]
                a = Xb
                for i, (w, b) in enumerate(zip(self._w, self._b)):
                    z = a @ w + b
                    if i < len(self._w) - 1:
                        a = self._relu(z)
                    else:
                        a = self._sigmoid(z)
                    activations.append(a)
                pred = activations[-1].squeeze()
                # Backward (output layer)
                delta = (pred - yb) / len(batch)
                for i in reversed(range(len(self._w))):
                    dw = activations[i].T @ delta.reshape(-1, 1)
                    db = delta.sum(axis=0)
                    self._w[i] -= lr * dw
                    self._b[i] -= lr * db
                    if i > 0:
                        delta = (delta.reshape(-1, 1) @ self._w[i].T).squeeze()
                        delta *= (activations[i] > 0).astype(float)

    # --- Public interface ---
    def train(self, X: np.ndarray, y: np.ndarray):
        # Normalize
        self._mean = X.mean(axis=0)
        self._std  = X.std(axis=0) + 1e-8
        Xn = (X - self._mean) / self._std
        self._input_dim = X.shape[1]

        if self.use_tf:
            self.model = self._build_tf_model(Xn.shape[1])
            self.history = self.model.fit(
                Xn, y.astype(np.float32),
                epochs=self.epochs,
                batch_size=64,
                validation_split=0.15,
                verbose=0
            )
        else:
            self._train_numpy(Xn, y)
        self.trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trained:
            return np.zeros(len(X), dtype=int)
        Xn = (X - self._mean) / self._std
        if self.use_tf:
            proba = self.model.predict(Xn, verbose=0).squeeze()
        else:
            proba = self._forward_numpy(Xn)
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.trained:
            return np.ones((len(X), 2)) * 0.5
        Xn = (X - self._mean) / self._std
        if self.use_tf:
            p1 = self.model.predict(Xn, verbose=0).squeeze()
        else:
            p1 = self._forward_numpy(Xn)
        return np.column_stack([1 - p1, p1])

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        from sklearn.metrics import (accuracy_score, precision_score,
                                     recall_score, f1_score, confusion_matrix)
        if not self.trained:
            return {}
        y_pred = self.predict(X)
        return {
            'accuracy':  round(accuracy_score(y, y_pred) * 100, 2),
            'precision': round(precision_score(y, y_pred, zero_division=0) * 100, 2),
            'recall':    round(recall_score(y, y_pred, zero_division=0) * 100, 2),
            'f1':        round(f1_score(y, y_pred, zero_division=0) * 100, 2),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
        }


# ============================================================
# SECTION 5 — ML + ANN ENSEMBLE
# ============================================================

class EnsemblePredictor:
    """
    Combines ML (Random Forest) + ANN predictions via soft voting.
    ML provides probability estimates, ANN refines with neural features.
    """

    def __init__(self, ml_weight: float = 0.45, ann_weight: float = 0.55):
        self.ml  = MLPredictor()
        self.ann = ANNPredictor()
        self.ml_weight  = ml_weight
        self.ann_weight = ann_weight
        self.trained = False

    def train(self, X: np.ndarray, y: np.ndarray):
        self.ml.train(X, y)
        self.ann.train(X, y)
        self.trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trained:
            return np.zeros(len(X), dtype=int)
        ml_proba  = self.ml.predict_proba(X)[:, 1]
        ann_proba = self.ann.predict_proba(X)[:, 1]
        combined  = self.ml_weight * ml_proba + self.ann_weight * ann_proba
        return (combined >= 0.5).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        from sklearn.metrics import (accuracy_score, precision_score,
                                     recall_score, f1_score, confusion_matrix)
        if not self.trained:
            return {}
        y_pred = self.predict(X)
        return {
            'accuracy':  round(accuracy_score(y, y_pred) * 100, 2),
            'precision': round(precision_score(y, y_pred, zero_division=0) * 100, 2),
            'recall':    round(recall_score(y, y_pred, zero_division=0) * 100, 2),
            'f1':        round(f1_score(y, y_pred, zero_division=0) * 100, 2),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
        }


# ============================================================
# SECTION 6 — TRAFFIC FLOW FORECASTER (LSTM / RF / Naive)
# ============================================================

class FlowForecaster:
    def __init__(self, lookback: int = 12, horizon: int = 6):
        self.lookback = lookback
        self.horizon  = horizon
        self.model    = None
        self.mode     = None

    def fit(self, series: List[float], epochs: int = 10):
        series = np.array(series, dtype=np.float32)
        if len(series) < self.lookback + self.horizon + 10:
            self.mode = 'naive'
            return
        X, y = [], []
        for i in range(len(series) - self.lookback - self.horizon + 1):
            X.append(series[i:i+self.lookback])
            y.append(series[i+self.lookback:i+self.lookback+self.horizon])
        X = np.array(X).reshape(-1, self.lookback, 1)
        y = np.array(y)
        if _TF is not None:
            try:
                tf = _TF
                self.model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(self.lookback, 1)),
                    tf.keras.layers.LSTM(32),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dense(self.horizon)
                ])
                self.model.compile(optimizer='adam', loss='mse')
                self.model.fit(X, y, epochs=epochs, batch_size=16, verbose=0)
                self.mode = 'lstm'
                return
            except Exception:
                self.model = None
        if _SKLEARN is not None:
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=50)
            self.model.fit(X.reshape(X.shape[0], -1), y.mean(axis=1))
            self.mode = 'rf'
        else:
            self.mode = 'naive'

    def predict(self, recent: List[float]) -> List[float]:
        s = np.array(recent[-self.lookback:], dtype=np.float32)
        if len(s) < self.lookback:
            s = np.concatenate([np.zeros(self.lookback - len(s)), s])
        if self.mode == 'lstm' and self.model:
            return self.model.predict(s.reshape(1, self.lookback, 1), verbose=0)[0].tolist()
        elif self.mode == 'rf' and self.model:
            v = float(max(0, self.model.predict(s.reshape(1, -1))[0]))
            return [v] * self.horizon
        else:
            return [float(s[-1])] * self.horizon


# ============================================================
# SECTION 7 — DQN RL SIGNAL CONTROLLER
# ============================================================

@dataclass
class TLState:
    q_ns:  int
    q_ew:  int
    phase: int


class SimpleIntersectionSim:
    def __init__(self, sat_flow=4, arr_ns=2.0, arr_ew=2.0):
        self.sat_flow = sat_flow
        self.arr_ns   = arr_ns
        self.arr_ew   = arr_ew
        self.reset()

    def reset(self) -> TLState:
        self.q_ns, self.q_ew, self.phase = 10, 10, 0
        return TLState(self.q_ns, self.q_ew, self.phase)

    def step(self, action: int) -> Tuple[TLState, float]:
        penalty = 0.0
        if action == 1:
            self.phase = 1 - self.phase
            penalty = 1.0
        self.q_ns += np.random.poisson(self.arr_ns)
        self.q_ew += np.random.poisson(self.arr_ew)
        if self.phase == 0:
            self.q_ns -= min(self.q_ns, self.sat_flow)
            self.q_ew -= min(self.q_ew, 1)
        else:
            self.q_ew -= min(self.q_ew, self.sat_flow)
            self.q_ns -= min(self.q_ns, 1)
        reward = -(self.q_ns + self.q_ew) - penalty
        return TLState(self.q_ns, self.q_ew, self.phase), reward


class DQNAgent:
    def __init__(self, state_dim=3, action_dim=2, gamma=0.95, lr=1e-3):
        self.use_torch  = _TORCH is not None
        self.gamma      = gamma
        self.action_dim = action_dim
        self.memory     = []
        self.max_mem    = 5000
        self.batch      = 64
        self.eps        = 1.0
        self.eps_min    = 0.05
        self.eps_decay  = 0.995

        if self.use_torch:
            import torch.nn as nn
            self.device = _TORCH.device('cuda' if _TORCH.cuda.is_available() else 'cpu')
            class Net(nn.Module):
                def __init__(self, s, a):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(s, 64), nn.ReLU(),
                        nn.Linear(64, 64), nn.ReLU(),
                        nn.Linear(64, a)
                    )
                def forward(self, x): return self.net(x)
            self.q      = Net(state_dim, action_dim).to(self.device)
            self.opt    = _TORCH.optim.Adam(self.q.parameters(), lr=lr)
            self.loss_fn = _TORCH.nn.MSELoss()
        else:
            self.device = None

    def select_action(self, state: TLState) -> int:
        if self.use_torch:
            if np.random.rand() < self.eps:
                return np.random.randint(self.action_dim)
            s = _TORCH.tensor([state.q_ns, state.q_ew, state.phase],
                               dtype=_TORCH.float32).to(self.device)
            with _TORCH.no_grad():
                return int(_TORCH.argmax(self.q(s)).item())
        else:
            if state.q_ns > state.q_ew * 1.2:
                return 0 if state.phase == 0 else 1
            if state.q_ew > state.q_ns * 1.2:
                return 1 if state.phase == 1 else 1
            return 0

    def remember(self, s, a, r, s2, done=False):
        if not self.use_torch: return
        self.memory.append((s, a, r, s2, done))
        if len(self.memory) > self.max_mem:
            self.memory = self.memory[-self.max_mem:]

    def learn(self):
        if not self.use_torch or len(self.memory) < self.batch: return
        idx   = np.random.choice(len(self.memory), self.batch, replace=False)
        batch = [self.memory[i] for i in idx]
        s  = _TORCH.tensor([x[0] for x in batch], dtype=_TORCH.float32).to(self.device)
        a  = _TORCH.tensor([x[1] for x in batch], dtype=_TORCH.long).to(self.device)
        r  = _TORCH.tensor([x[2] for x in batch], dtype=_TORCH.float32).to(self.device)
        s2 = _TORCH.tensor([x[3] for x in batch], dtype=_TORCH.float32).to(self.device)
        d  = _TORCH.tensor([x[4] for x in batch], dtype=_TORCH.float32).to(self.device)
        qvals  = self.q(s).gather(1, a.view(-1,1)).squeeze()
        with _TORCH.no_grad():
            qnext = self.q(s2).max(1)[0]
        target = r + self.gamma * qnext * (1 - d)
        loss   = self.loss_fn(qvals, target)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.eps = max(self.eps_min, self.eps * self.eps_decay)


# ============================================================
# SECTION 8 — MQTT (optional)
# ============================================================

class MQTTHook:
    def __init__(self, host='localhost', port=1883, topic='itms/metrics'):
        self.client = None
        self.topic  = topic
        if _PAHO is not None:
            try:
                self.client = _PAHO.Client()
                self.client.connect(host, port, 60)
                self.client.loop_start()
            except Exception:
                self.client = None

    def publish(self, payload: Dict):
        if self.client is None: return
        try:
            self.client.publish(self.topic, json.dumps(payload))
        except Exception:
            pass


# ============================================================
# STREAMLIT APPLICATION
# ============================================================

st.set_page_config(page_title="AI Traffic Signal Control", layout="wide")
st.title("🚦 AI-Powered Adaptive Traffic Signal Control System")
st.caption("Asad Irfan (65117) · Mohsin Khan (62876)")

# --- Session state init ---
for key, default in [
    ('df', None), ('X_train', None), ('X_test', None),
    ('y_train', None), ('y_test', None), ('features', None),
    ('ml_model', MLPredictor()), ('ann_model', ANNPredictor()),
    ('ensemble', EnsemblePredictor()), ('forecaster', FlowForecaster()),
    ('sim', SimpleIntersectionSim()), ('agent', DQNAgent()),
    ('mqtt', MQTTHook()), ('ml_metrics', None),
    ('ann_metrics', None), ('ens_metrics', None), ('trained', False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# SIDEBAR — Data Source
# ============================================================
with st.sidebar:
    st.header("⚙️ Settings")
    st.subheader("1. Dataset")
    data_source = st.radio("Data Source", ["Use Synthetic Dataset", "Upload Kaggle CSV"])

    if data_source == "Upload Kaggle CSV":
        uploaded = st.file_uploader(
            "Upload traffic CSV from Kaggle",
            type=["csv"],
            help="Supports most Kaggle traffic datasets. Columns are auto-detected."
        )
        if uploaded:
            try:
                df = load_kaggle_csv(uploaded)
                st.session_state.df = df
                st.success(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
                st.write("**Detected columns:**", list(df.columns))
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
    else:
        n_rows = st.slider("Synthetic rows", 500, 5000, 2000, 500)
        if st.button("Generate Synthetic Dataset"):
            st.session_state.df = generate_synthetic_dataset(n_rows)
            st.success(f"✅ Generated {n_rows} synthetic traffic records")

    st.subheader("2. Train/Test Split")
    test_size = st.slider("Test set %", 10, 40, 20, 5) / 100

    st.subheader("3. Model Settings")
    ann_epochs   = st.slider("ANN epochs",   10, 100, 30, 10)
    rf_trees     = st.slider("RF n_estimators", 50, 300, 100, 50)
    ml_weight    = st.slider("Ensemble ML weight",  0.1, 0.9, 0.45, 0.05)

    st.subheader("4. Optional")
    st.caption("MQTT auto-enabled if paho-mqtt is installed.")

    train_btn = st.button("🚀 Train All Models", type="primary",
                          disabled=(st.session_state.df is None))


# ============================================================
# TRAINING PIPELINE
# ============================================================

if train_btn and st.session_state.df is not None:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df = st.session_state.df
    X, y, features = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    st.session_state.X_train  = X_train
    st.session_state.X_test   = X_test
    st.session_state.y_train  = y_train
    st.session_state.y_test   = y_test
    st.session_state.features = features

    with st.spinner("Training Random Forest (ML)..."):
        ml = MLPredictor(n_estimators=rf_trees)
        ml.train(X_train, y_train)
        st.session_state.ml_model   = ml
        st.session_state.ml_metrics = ml.evaluate(X_test, y_test)

    with st.spinner("Training ANN..."):
        ann = ANNPredictor(epochs=ann_epochs)
        ann.train(X_train, y_train)
        st.session_state.ann_model   = ann
        st.session_state.ann_metrics = ann.evaluate(X_test, y_test)

    with st.spinner("Training ML + ANN Ensemble..."):
        ens = EnsemblePredictor(ml_weight=ml_weight, ann_weight=1-ml_weight)
        ens.ml  = ml    # reuse already trained models
        ens.ann = ann
        ens.trained = True
        st.session_state.ensemble    = ens
        st.session_state.ens_metrics = ens.evaluate(X_test, y_test)

    with st.spinner("Training LSTM Forecaster..."):
        forecaster = FlowForecaster()
        forecaster.fit(df['vehicle_count'].tolist(), epochs=10)
        st.session_state.forecaster = forecaster

    st.session_state.trained = True
    st.success("✅ All models trained successfully!")


# ============================================================
# MAIN TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset Explorer",
    "🤖 ML vs ANN Accuracy",
    "🧠 Ensemble Predictor",
    "📈 Traffic Forecasting",
    "🎮 RL Signal Simulator",
])


# ─── TAB 1: Dataset Explorer ───────────────────────────────
with tab1:
    if st.session_state.df is None:
        st.info("👈 Load or generate a dataset from the sidebar first.")
    else:
        df = st.session_state.df
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records",    f"{len(df):,}")
        col2.metric("Avg Vehicle Count", f"{df['vehicle_count'].mean():.1f}")
        col3.metric("Avg Waiting Time",  f"{df['waiting_time'].mean():.1f}s")
        col4.metric("Phase 1 (EW green)",f"{(df['signal_phase']==1).mean()*100:.1f}%")

        st.subheader("Sample Data")
        st.dataframe(df.head(50), use_container_width=True)

        st.subheader("Vehicle Count by Hour")
        hourly = df.groupby('hour')['vehicle_count'].mean().reset_index()
        st.line_chart(hourly.set_index('hour'))

        st.subheader("Congestion Level Distribution")
        if 'congestion_level' in df.columns:
            cong = df['congestion_level'].value_counts().sort_index()
            st.bar_chart(cong)

        st.subheader("Queue Length: NS vs EW")
        queue_df = df[['queue_ns', 'queue_ew']].head(200)
        st.line_chart(queue_df)


# ─── TAB 2: ML vs ANN Accuracy Comparison ──────────────────
with tab2:
    st.subheader("📊 Accuracy Comparison: ML alone vs ML + ANN")

    if not st.session_state.trained:
        st.info("Train the models first using the sidebar button.")
    else:
        ml_m  = st.session_state.ml_metrics
        ann_m = st.session_state.ann_metrics
        ens_m = st.session_state.ens_metrics

        # Key metrics table
        metrics_df = pd.DataFrame({
            'Model':     ['ML (Random Forest)', 'ANN', 'ML + ANN Ensemble'],
            'Accuracy %':  [ml_m['accuracy'],  ann_m['accuracy'],  ens_m['accuracy']],
            'Precision %': [ml_m['precision'], ann_m['precision'], ens_m['precision']],
            'Recall %':    [ml_m['recall'],    ann_m['recall'],    ens_m['recall']],
            'F1 Score %':  [ml_m['f1'],        ann_m['f1'],        ens_m['f1']],
        })
        st.dataframe(metrics_df.set_index('Model'), use_container_width=True)

        # Visual accuracy bar chart
        st.subheader("Accuracy Bar Chart")
        acc_chart = pd.DataFrame({
            'Accuracy (%)': [ml_m['accuracy'], ann_m['accuracy'], ens_m['accuracy']]
        }, index=['ML Only', 'ANN Only', 'ML + ANN'])
        st.bar_chart(acc_chart)

        # Improvement callout
        improvement = ens_m['accuracy'] - ml_m['accuracy']
        if improvement > 0:
            st.success(f"✅ ML + ANN Ensemble improves accuracy by **{improvement:.2f}%** over ML alone.")
        elif improvement == 0:
            st.info("ML alone and the ensemble achieve the same accuracy on this dataset.")
        else:
            st.warning(f"ML alone is {abs(improvement):.2f}% more accurate on this dataset. "
                       "Try adjusting ensemble weights.")

        # Side-by-side confusion matrices
        st.subheader("Confusion Matrices")
        c1, c2, c3 = st.columns(3)
        for col, label, m in zip([c1, c2, c3],
                                  ['ML Only', 'ANN Only', 'ML + ANN'],
                                  [ml_m, ann_m, ens_m]):
            cm = np.array(m['confusion_matrix'])
            col.write(f"**{label}**")
            cm_df = pd.DataFrame(cm,
                                  index=['Actual NS', 'Actual EW'],
                                  columns=['Pred NS', 'Pred EW'])
            col.dataframe(cm_df)

        # Feature importances (ML only)
        if st.session_state.ml_model.feature_importances_ is not None:
            st.subheader("Feature Importances (Random Forest)")
            fi = pd.Series(
                st.session_state.ml_model.feature_importances_,
                index=st.session_state.features
            ).sort_values(ascending=False)
            st.bar_chart(fi)

        # ANN training history (if TF was used)
        ann_hist = st.session_state.ann_model.history
        if ann_hist is not None and hasattr(ann_hist, 'history'):
            st.subheader("ANN Training History")
            hist_df = pd.DataFrame({
                'Train Accuracy': ann_hist.history.get('accuracy', []),
                'Val Accuracy':   ann_hist.history.get('val_accuracy', []),
                'Train Loss':     ann_hist.history.get('loss', []),
                'Val Loss':       ann_hist.history.get('val_loss', []),
            })
            c1, c2 = st.columns(2)
            c1.line_chart(hist_df[['Train Accuracy', 'Val Accuracy']])
            c2.line_chart(hist_df[['Train Loss', 'Val Loss']])


# ─── TAB 3: Ensemble Live Predictor ────────────────────────
with tab3:
    st.subheader("🧠 Real-Time Signal Phase Prediction (ML + ANN Ensemble)")

    if not st.session_state.trained:
        st.info("Train the models first.")
    else:
        st.markdown("Adjust the sliders to simulate real-time traffic conditions:")
        c1, c2 = st.columns(2)
        with c1:
            hour      = st.slider("Hour of day",    0, 23, 8)
            dow       = st.slider("Day of week",    0, 6,  1)
            vehicles  = st.slider("Vehicle count",  0, 60, 25)
            speed     = st.slider("Avg speed (km/h)", 5, 80, 35)
        with c2:
            queue_ns  = st.slider("Queue NS (vehicles)", 0, 40, 12)
            queue_ew  = st.slider("Queue EW (vehicles)", 0, 40, 8)
            wait_time = st.slider("Avg waiting time (s)", 0, 200, 60)
            cong      = st.slider("Congestion level (0-3)", 0, 3, 1)

        features = st.session_state.features
        feat_map = {
            'hour': hour, 'day_of_week': dow, 'vehicle_count': vehicles,
            'speed': speed, 'queue_ns': queue_ns, 'queue_ew': queue_ew,
            'waiting_time': wait_time, 'congestion_level': cong,
        }
        X_live = np.array([[feat_map.get(f, 0) for f in features]], dtype=np.float32)

        ml_pred  = st.session_state.ml_model.predict(X_live)[0]
        ann_pred = st.session_state.ann_model.predict(X_live)[0]
        ens_pred = st.session_state.ensemble.predict(X_live)[0]

        ml_conf  = st.session_state.ml_model.predict_proba(X_live)[0]
        ann_conf = st.session_state.ann_model.predict_proba(X_live)[0]
        ens_prob = (ml_conf * st.session_state.ensemble.ml_weight +
                    ann_conf * st.session_state.ensemble.ann_weight)

        phase_label = {0: "🟢 NS Green (North-South)", 1: "🟢 EW Green (East-West)"}

        st.divider()
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("ML Prediction",       phase_label[ml_pred],
                   f"Confidence: {max(ml_conf)*100:.1f}%")
        pc2.metric("ANN Prediction",      phase_label[ann_pred],
                   f"Confidence: {max(ann_conf)*100:.1f}%")
        pc3.metric("Ensemble (ML + ANN)", phase_label[ens_pred],
                   f"Confidence: {max(ens_prob)*100:.1f}%")

        # Publish to MQTT
        st.session_state.mqtt.publish({
            'ts': int(time.time()),
            'ensemble_phase': int(ens_pred),
            'vehicle_count': vehicles,
            'queue_ns': queue_ns,
            'queue_ew': queue_ew,
        })

        st.divider()
        st.subheader("Batch Prediction on Test Set")
        if st.button("Run batch prediction on test data"):
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            ens_preds = st.session_state.ensemble.predict(X_test)
            result_df = pd.DataFrame({
                'Actual Phase': y_test[:100],
                'Ensemble Prediction': ens_preds[:100],
                'Correct': (y_test[:100] == ens_preds[:100]).astype(int)
            })
            st.dataframe(result_df, use_container_width=True)
            acc = result_df['Correct'].mean() * 100
            st.metric("Batch Accuracy (first 100)", f"{acc:.1f}%")


# ─── TAB 4: Traffic Flow Forecasting ───────────────────────
with tab4:
    st.subheader("📈 Traffic Flow Forecasting (LSTM / RF fallback)")

    if st.session_state.df is None:
        st.info("Load a dataset first.")
    else:
        df = st.session_state.df
        series = df['vehicle_count'].tolist()
        forecaster = st.session_state.forecaster

        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("(Re)train Forecaster"):
                with st.spinner("Training..."):
                    forecaster.fit(series, epochs=10)
                    st.session_state.forecaster = forecaster
                st.success(f"Trained ({forecaster.mode.upper()})")

        preds = forecaster.predict(series)
        pred_df = pd.DataFrame({
            'Step':           list(range(1, len(preds)+1)),
            'Predicted Count': [round(p, 1) for p in preds]
        }).set_index('Step')
        col1.write(f"**Next {len(preds)} steps forecast ({forecaster.mode}):**")
        col1.dataframe(pred_df.T)

        # Historical + forecast chart
        recent = series[-60:]
        chart_df = pd.DataFrame({
            'Historical': recent + [None]*len(preds),
            'Forecast':   [None]*len(recent) + preds
        })
        st.line_chart(chart_df)

        # Hourly pattern
        st.subheader("Average Vehicle Count by Hour")
        hourly = df.groupby('hour')['vehicle_count'].mean()
        st.bar_chart(hourly)


# ─── TAB 5: RL Signal Simulator ────────────────────────────
with tab5:
    st.subheader("🎮 DQN Reinforcement Learning Signal Simulator")
    st.caption("Trains a DQN agent on a simulated intersection. "
               "Uses PyTorch if available, else heuristic fallback.")

    c1, c2, c3 = st.columns(3)
    arr_ns = c1.slider("Arrival rate NS (vehicles/step)", 0.5, 5.0, 2.0, 0.5)
    arr_ew = c2.slider("Arrival rate EW (vehicles/step)", 0.5, 5.0, 2.0, 0.5)
    steps  = c3.slider("Simulation steps", 50, 500, 150, 50)

    compare_static = st.checkbox("Also run static timer baseline for comparison", value=True)

    if st.button("▶ Run RL Simulation"):
        sim   = SimpleIntersectionSim(arr_ns=arr_ns, arr_ew=arr_ew)
        agent = st.session_state.agent
        s = sim.reset()

        rl_rewards, rl_queues = [], []
        for t in range(steps):
            a = agent.select_action(s)
            s2, r = sim.step(a)
            if agent.use_torch:
                agent.remember([s.q_ns, s.q_ew, s.phase], a, r,
                               [s2.q_ns, s2.q_ew, s2.phase])
                agent.learn()
            rl_rewards.append(r)
            rl_queues.append(s2.q_ns + s2.q_ew)
            s = s2
        st.session_state.agent = agent

        results = {
            'RL Agent':    {'rewards': rl_rewards, 'queues': rl_queues}
        }

        if compare_static:
            # Static: switch every 10 steps
            sim2 = SimpleIntersectionSim(arr_ns=arr_ns, arr_ew=arr_ew)
            s2_ = sim2.reset()
            static_rewards, static_queues = [], []
            for t in range(steps):
                action = 1 if t % 10 == 0 else 0
                s2_, r = sim2.step(action)
                static_rewards.append(r)
                static_queues.append(s2_.q_ns + s2_.q_ew)
            results['Static Timer'] = {'rewards': static_rewards, 'queues': static_queues}

        # Charts
        queue_chart = pd.DataFrame({k: v['queues'] for k, v in results.items()})
        reward_chart = pd.DataFrame({k: v['rewards'] for k, v in results.items()})

        st.subheader("Total Queue Length over Time")
        st.line_chart(queue_chart)

        st.subheader("Reward over Time")
        st.line_chart(reward_chart)

        # Summary
        st.subheader("Summary")
        sum_rows = []
        for name, data in results.items():
            sum_rows.append({
                'Controller': name,
                'Avg Queue': round(np.mean(data['queues']), 2),
                'Avg Reward': round(np.mean(data['rewards']), 2),
                'Total Reward': round(sum(data['rewards']), 1),
            })
        st.dataframe(pd.DataFrame(sum_rows).set_index('Controller'),
                     use_container_width=True)

        if compare_static:
            rl_avg    = np.mean(rl_queues)
            stat_avg  = np.mean(static_queues)
            reduction = (stat_avg - rl_avg) / stat_avg * 100
            if reduction > 0:
                st.success(f"✅ RL agent reduces average queue by **{reduction:.1f}%** vs static timer.")
            else:
                st.info(f"Static timer performed better by {abs(reduction):.1f}% on this run. "
                        "Train more steps for the RL agent to converge.")

# Footer
st.divider()
st.caption("AI-Powered Adaptive Traffic Signal Control · "
           "Technologies: YOLOv8 · OpenCV · LSTM · Random Forest · ANN · DQN RL · "
           "TensorFlow · PyTorch · Kaggle Dataset · Streamlit")
