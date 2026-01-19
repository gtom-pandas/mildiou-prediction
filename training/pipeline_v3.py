"""
================================================================================
MILDIOU AI PREDICTION PIPELINE - VERSION 3.0
================================================================================
Agricultural Machine Learning System for Downy Mildew Risk Assessment
Target Platform: Arduino UNO R4 WiFi (Edge Computing / TinyML)

Author: Graci Tom - Institut Agro Dijon
License: MIT

Key Changes in V3:
- Replaced wind_speed with meanpressure as predictor variable
- Added pressure-based features (barometric trends correlate with rain events)
- Pressure drop detection as early warning indicator
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """
    Central configuration class containing all hyperparameters and thresholds.
    Modify these values to tune model behavior and epidemiological assumptions.
    """
    
    # --- Data Sources ---
    TRAIN_FILE = "/kaggle/input/meteofrance/meteofranceTrain.csv" #free access open data Météo FRANCE
    TEST_FILE = "/kaggle/input/meteofrance/meteofranceTest.csv" #free access open data Météo FRANCE
    
    # --- Epidemiological Parameters (Plasmopara viticola) ---
    # Temperature range favorable for sporulation (Celsius)
    TEMP_MIN_FAVORABLE = 15.0
    TEMP_MAX_FAVORABLE = 30.0
    TEMP_OPTIMAL_MIN = 23.0
    TEMP_OPTIMAL_MAX = 25.0
    
    # Relative humidity thresholds (%)
    HUMIDITY_THRESHOLD = 70.0
    HUMIDITY_CRITICAL = 90.0
    
    # --- Atmospheric Pressure Thresholds (hPa) ---
    # Standard sea-level pressure: ~1013 hPa
    # Low pressure systems (<1010 hPa) often precede precipitation events
    PRESSURE_NORMAL = 1013.0
    PRESSURE_LOW = 1008.0
    PRESSURE_DROP_THRESHOLD = 5.0  # 5 hPa drop over 3 days triggers alert
    
    # --- Data Augmentation (Sensor Noise Simulation) ---
    # Simulates DHT11 sensor inaccuracies for model robustness
    NOISE_TEMP_STD = 2.0       # ±2°C (DHT11 typical error)
    NOISE_HUM_STD = 5.0        # ±5% RH (DHT11 typical error)
    NOISE_PRESSURE_STD = 1.0   # ±1 hPa (LPS25 higher precision)
    AUGMENTATION_FACTOR = 3    # Number of augmented copies per sample
    
    # --- Neural Network Architecture ---
    HIDDEN_LAYERS = [20, 10]   # MLP hidden layer dimensions
    OUTPUT_SIZE = 3            # Classification: [Low, Medium, High]
    
    # --- Training Hyperparameters ---
    LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    EPOCHS = 500
    BATCH_SIZE = 32
    
    # --- Risk Classification Thresholds ---
    SCORE_THRESHOLD_MEDIUM = 3
    SCORE_THRESHOLD_HIGH = 5


# ==============================================================================
# STEP 1: DATA LOADING
# ==============================================================================

def load_data():
    """
    Load and merge training/test datasets from CSV files.
    
    Returns:
        pd.DataFrame: Combined dataset with 'split' column indicating origin.
    """
    print("=" * 60)
    print("STEP 1: DATA LOADING")
    print("=" * 60)
    
    df_train = pd.read_csv(Config.TRAIN_FILE, parse_dates=['date'])
    df_test = pd.read_csv(Config.TEST_FILE, parse_dates=['date'])
    
    print(f"\n[INFO] Training set: {len(df_train)} days "
          f"({df_train['date'].min().date()} to {df_train['date'].max().date()})")
    print(f"[INFO] Test set: {len(df_test)} days "
          f"({df_test['date'].min().date()} to {df_test['date'].max().date()})")
    
    df_train['split'] = 'train'
    df_test['split'] = 'test'
    df = pd.concat([df_train, df_test], ignore_index=True)
    df = df.sort_values('date').reset_index(drop=True)
    
    # Handle pressure outliers (dataset contains some erroneous values)
    median_pressure = df.loc[
        (df['meanpressure'] > 900) & (df['meanpressure'] < 1100), 
        'meanpressure'
    ].median()
    df.loc[df['meanpressure'] < 900, 'meanpressure'] = median_pressure
    df.loc[df['meanpressure'] > 1100, 'meanpressure'] = median_pressure
    
    print(f"\n[STATS] Temperature: {df['meantemp'].min():.1f}C to "
          f"{df['meantemp'].max():.1f}C (mean: {df['meantemp'].mean():.1f}C)")
    print(f"[STATS] Humidity: {df['humidity'].min():.1f}% to "
          f"{df['humidity'].max():.1f}% (mean: {df['humidity'].mean():.1f}%)")
    print(f"[STATS] Pressure: {df['meanpressure'].min():.1f} to "
          f"{df['meanpressure'].max():.1f} hPa (mean: {df['meanpressure'].mean():.1f} hPa)")
    
    return df


# ==============================================================================
# STEP 2: TEMPORAL FEATURE ENGINEERING
# ==============================================================================

def create_temporal_features(df):
    """
    Generate time-series features from raw meteorological data.
    
    Features include:
    - Trend calculations (derivatives over 1-3 day windows)
    - Moving averages (5-day and 7-day windows)
    - Statistical measures (standard deviation, min/max)
    - Consecutive day counters for favorable conditions
    - Cyclical encoding for seasonality
    
    Args:
        df: DataFrame with raw weather data
        
    Returns:
        pd.DataFrame: Enhanced dataset with temporal features
    """
    print("\n" + "=" * 60)
    print("STEP 2: TEMPORAL FEATURE ENGINEERING")
    print("=" * 60)
    
    df = df.copy()
    
    # --- 2.1 Trend Calculations (First Derivatives) ---
    print("\n[2.1] Computing trend features...")
    
    df['temp_trend_3d'] = df['meantemp'].diff(periods=3)
    df['humidity_trend_3d'] = df['humidity'].diff(periods=3)
    df['pressure_trend_3d'] = df['meanpressure'].diff(periods=3)
    df['pressure_trend_1d'] = df['meanpressure'].diff(periods=1)
    
    # --- 2.2 Moving Averages ---
    print("[2.2] Computing moving averages...")
    
    df['temp_ma5'] = df['meantemp'].rolling(window=5, min_periods=1).mean()
    df['temp_ma7'] = df['meantemp'].rolling(window=7, min_periods=1).mean()
    df['humidity_ma5'] = df['humidity'].rolling(window=5, min_periods=1).mean()
    df['humidity_ma7'] = df['humidity'].rolling(window=7, min_periods=1).mean()
    df['pressure_ma5'] = df['meanpressure'].rolling(window=5, min_periods=1).mean()
    df['pressure_ma7'] = df['meanpressure'].rolling(window=7, min_periods=1).mean()
    
    # --- 2.3 Statistical Features ---
    print("[2.3] Computing statistical features...")
    
    df['temp_std7'] = df['meantemp'].rolling(window=7, min_periods=1).std()
    df['humidity_std7'] = df['humidity'].rolling(window=7, min_periods=1).std()
    df['pressure_std7'] = df['meanpressure'].rolling(window=7, min_periods=1).std()
    df['humidity_max7'] = df['humidity'].rolling(window=7, min_periods=1).max()
    df['humidity_min7'] = df['humidity'].rolling(window=7, min_periods=1).min()
    df['pressure_min7'] = df['meanpressure'].rolling(window=7, min_periods=1).min()
    
    # --- 2.4 Pressure-Based Risk Indicators ---
    print("[2.4] Computing pressure indicators...")
    
    df['low_pressure'] = (df['meanpressure'] < Config.PRESSURE_LOW).astype(int)
    df['pressure_dropping'] = (
        df['pressure_trend_3d'] < -Config.PRESSURE_DROP_THRESHOLD
    ).astype(int)
    df['pressure_anomaly'] = df['meanpressure'] - Config.PRESSURE_NORMAL
    
    # --- 2.5 Consecutive Day Counters ---
    print("[2.5] Computing consecutive day counters...")
    
    df['temp_favorable'] = (
        (df['meantemp'] >= Config.TEMP_MIN_FAVORABLE) & 
        (df['meantemp'] <= Config.TEMP_MAX_FAVORABLE)
    ).astype(int)
    df['consecutive_temp_favorable'] = calculate_consecutive(df['temp_favorable'])
    
    df['humidity_high'] = (df['humidity'] >= Config.HUMIDITY_THRESHOLD).astype(int)
    df['consecutive_humidity_high'] = calculate_consecutive(df['humidity_high'])
    
    df['both_favorable'] = (df['temp_favorable'] & df['humidity_high']).astype(int)
    df['consecutive_both'] = calculate_consecutive(df['both_favorable'])
    df['consecutive_low_pressure'] = calculate_consecutive(df['low_pressure'])
    
    # --- 2.6 Cumulative Risk Accumulator ---
    print("[2.6] Computing risk accumulation...")
    df['risk_accumulator'] = calculate_risk_accumulator(df)
    
    # --- 2.7 Cyclical Time Encoding ---
    print("[2.7] Computing cyclical features...")
    
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['high_risk_season'] = ((df['month'] >= 3) & (df['month'] <= 10)).astype(int)
    
    # --- 2.8 Feature Interactions ---
    print("[2.8] Computing feature interactions...")
    
    df['temp_humidity_interaction'] = (df['meantemp'] / 40) * (df['humidity'] / 100)
    df['humidity_pressure_interaction'] = (
        (df['humidity'] / 100) * (1 - (df['meanpressure'] - 990) / 40)
    )
    df['humidity_pressure_interaction'] = df['humidity_pressure_interaction'].clip(0, 1)
    
    df = df.fillna(0)
    
    print(f"\n[SUCCESS] Feature engineering complete")
    
    return df


def calculate_consecutive(series):
    """
    Count consecutive days where condition is True.
    
    Args:
        series: Boolean pandas Series
        
    Returns:
        np.ndarray: Consecutive day count at each position
    """
    result = np.zeros(len(series))
    count = 0
    for i, val in enumerate(series):
        count = count + 1 if val else 0
        result[i] = count
    return result


def calculate_risk_accumulator(df):
    """
    Compute cumulative risk score with exponential decay.
    
    The accumulator increases when conditions favor pathogen development
    and decays exponentially when conditions are unfavorable.
    
    Args:
        df: DataFrame with meteorological features
        
    Returns:
        np.ndarray: Normalized risk accumulation values [0, 1]
    """
    risk = np.zeros(len(df))
    decay = 0.85  # 15% daily decay rate
    
    for i in range(len(df)):
        row = df.iloc[i]
        instant_risk = 0
        
        # Temperature component (0-0.30)
        temp = row['meantemp']
        if Config.TEMP_OPTIMAL_MIN <= temp <= Config.TEMP_OPTIMAL_MAX:
            instant_risk += 0.30
        elif Config.TEMP_MIN_FAVORABLE <= temp <= Config.TEMP_MAX_FAVORABLE:
            instant_risk += 0.15
        
        # Humidity component (0-0.35)
        humidity = row['humidity']
        if humidity >= Config.HUMIDITY_CRITICAL:
            instant_risk += 0.35
        elif humidity >= Config.HUMIDITY_THRESHOLD:
            instant_risk += 0.20
        
        # Pressure component (0-0.25)
        pressure = row['meanpressure']
        if pressure < Config.PRESSURE_LOW:
            instant_risk += 0.25
        elif pressure < Config.PRESSURE_NORMAL:
            instant_risk += 0.10
        
        # Pressure drop bonus (0-0.10)
        if i >= 3:
            pressure_drop = df.iloc[i-3]['meanpressure'] - pressure
            if pressure_drop > Config.PRESSURE_DROP_THRESHOLD:
                instant_risk += 0.10
        
        # Apply decay and accumulate
        risk[i] = risk[i-1] * decay + instant_risk if i > 0 else instant_risk
    
    # Normalize to [0, 1]
    max_risk = risk.max()
    if max_risk > 0:
        risk = risk / max_risk
    
    return risk


# ==============================================================================
# STEP 3: AUTOMATIC LABELING
# ==============================================================================

def create_labels(df):
    """
    Generate risk labels using domain-knowledge rules.
    
    Classification scheme:
    - 0: Low risk
    - 1: Medium risk  
    - 2: High risk
    
    Args:
        df: DataFrame with temporal features
        
    Returns:
        pd.DataFrame: Dataset with risk_score and risk_label columns
    """
    print("\n" + "=" * 60)
    print("STEP 3: AUTOMATIC LABELING")
    print("=" * 60)
    
    labels = np.zeros(len(df), dtype=int)
    scores = np.zeros(len(df))
    
    for i in range(len(df)):
        row = df.iloc[i]
        score = 0
        
        # Criterion 1: Risk accumulation level
        if row['risk_accumulator'] >= 0.6:
            score += 3
        elif row['risk_accumulator'] >= 0.4:
            score += 2
        elif row['risk_accumulator'] >= 0.25:
            score += 1
        
        # Criterion 2: Consecutive favorable conditions
        if row['consecutive_both'] >= 4:
            score += 3
        elif row['consecutive_both'] >= 2:
            score += 2
        elif row['consecutive_both'] >= 1:
            score += 1
        
        # Criterion 3: Humidity level
        if row['humidity'] >= Config.HUMIDITY_CRITICAL:
            score += 2
        elif row['humidity'] >= Config.HUMIDITY_THRESHOLD:
            score += 1
        
        # Criterion 4: Optimal temperature
        if Config.TEMP_OPTIMAL_MIN <= row['meantemp'] <= Config.TEMP_OPTIMAL_MAX:
            score += 1
        
        # Criterion 5: Low pressure
        if row['meanpressure'] < Config.PRESSURE_LOW:
            score += 2
        elif row['meanpressure'] < Config.PRESSURE_NORMAL - 3:
            score += 1
        
        # Criterion 6: Pressure dropping
        if row['pressure_dropping']:
            score += 1
        
        # Criterion 7: Humidity trend
        if row['humidity_trend_3d'] > 5:
            score += 1
        
        # Criterion 8: Seasonal multiplier
        if row['high_risk_season']:
            score = int(score * 1.2)
        
        scores[i] = score
        
        # Classify based on thresholds
        if score >= Config.SCORE_THRESHOLD_HIGH:
            labels[i] = 2
        elif score >= Config.SCORE_THRESHOLD_MEDIUM:
            labels[i] = 1
        else:
            labels[i] = 0
    
    df['risk_score'] = scores
    df['risk_label'] = labels
    df['risk_label'] = smooth_labels(df['risk_label'].values)
    
    # Display distribution
    print("\n[DISTRIBUTION] Label counts:")
    for label, name in [(0, 'LOW'), (1, 'MEDIUM'), (2, 'HIGH')]:
        count = (df['risk_label'] == label).sum()
        pct = count / len(df) * 100
        bar = '#' * int(pct / 5)
        print(f"   {name:8}: {count:5} ({pct:5.1f}%) {bar}")
    
    return df


def smooth_labels(labels):
    """
    Apply temporal smoothing to prevent rapid label oscillations.
    
    Args:
        labels: Array of integer labels
        
    Returns:
        np.ndarray: Smoothed labels
    """
    smoothed = labels.copy()
    for i in range(1, len(labels)):
        if labels[i] == 0 and smoothed[i-1] == 2:
            smoothed[i] = 1
        elif labels[i] == 2 and smoothed[i-1] == 0:
            smoothed[i] = 1
    return smoothed


# ==============================================================================
# STEP 4: DATA PREPARATION
# ==============================================================================

def prepare_training_data(df):
    """
    Prepare feature matrices and labels for neural network training.
    
    Operations:
    - Feature selection (25 features including pressure-based)
    - Train/test split based on 'split' column
    - Z-score normalization
    - One-hot encoding for labels
    
    Args:
        df: DataFrame with all features and labels
        
    Returns:
        dict: Contains X_train, X_test, y_train, y_test, normalization params
    """
    print("\n" + "=" * 60)
    print("STEP 4: DATA PREPARATION")
    print("=" * 60)
    
    # Feature selection (25 features)
    feature_columns = [
        # Raw measurements (3)
        'meantemp', 'humidity', 'meanpressure',
        
        # Trends (3)
        'temp_trend_3d', 'humidity_trend_3d', 'pressure_trend_3d',
        
        # Moving averages (4)
        'temp_ma5', 'humidity_ma5', 'pressure_ma5', 'humidity_ma7',
        
        # Statistical features (3)
        'temp_std7', 'humidity_max7', 'pressure_min7',
        
        # Pressure indicators (2)
        'pressure_anomaly', 'pressure_dropping',
        
        # Consecutive counters (4)
        'consecutive_temp_favorable', 'consecutive_humidity_high',
        'consecutive_both', 'consecutive_low_pressure',
        
        # Risk accumulator (1)
        'risk_accumulator',
        
        # Cyclical and seasonal (3)
        'day_sin', 'day_cos', 'high_risk_season',
        
        # Interactions (2)
        'temp_humidity_interaction', 'humidity_pressure_interaction'
    ]
    
    print(f"\n[INFO] Selected {len(feature_columns)} features:")
    for i, col in enumerate(feature_columns):
        print(f"   {i+1:2d}. {col}")
    
    # Split data
    train_mask = df['split'] == 'train'
    test_mask = df['split'] == 'test'
    
    X_train = df.loc[train_mask, feature_columns].values
    y_train = df.loc[train_mask, 'risk_label'].values
    X_test = df.loc[test_mask, feature_columns].values
    y_test = df.loc[test_mask, 'risk_label'].values
    
    # Z-score normalization
    feature_means = X_train.mean(axis=0)
    feature_stds = X_train.std(axis=0)
    feature_stds[feature_stds == 0] = 1  # Prevent division by zero
    
    X_train_norm = (X_train - feature_means) / feature_stds
    X_test_norm = (X_test - feature_means) / feature_stds
    
    # One-hot encoding
    num_classes = 3
    y_train_onehot = np.eye(num_classes)[y_train]
    y_test_onehot = np.eye(num_classes)[y_test]
    
    print(f"\n[INFO] Training samples: {X_train_norm.shape[0]}")
    print(f"[INFO] Test samples: {X_test_norm.shape[0]}")
    
    # Display class distribution
    print(f"\n[DISTRIBUTION] Training set:")
    for i, name in enumerate(['LOW', 'MEDIUM', 'HIGH']):
        count = (y_train == i).sum()
        print(f"   {name}: {count} ({count/len(y_train)*100:.1f}%)")
    
    print(f"\n[DISTRIBUTION] Test set:")
    for i, name in enumerate(['LOW', 'MEDIUM', 'HIGH']):
        count = (y_test == i).sum()
        print(f"   {name}: {count} ({count/len(y_test)*100:.1f}%)")
    
    return {
        'X_train': X_train_norm,
        'y_train': y_train_onehot,
        'y_train_labels': y_train,
        'X_test': X_test_norm,
        'y_test': y_test_onehot,
        'y_test_labels': y_test,
        'feature_columns': feature_columns,
        'feature_means': feature_means,
        'feature_stds': feature_stds
    }


def augment_data(X_train, y_train, feature_columns):
    """
    Augment training data with synthetic sensor noise.
    
    Simulates typical DHT11/LPS25 sensor errors to improve model
    robustness against real-world measurement inaccuracies.
    
    Args:
        X_train: Normalized training features
        y_train: One-hot encoded labels
        feature_columns: List of feature names
        
    Returns:
        tuple: (augmented_X, augmented_y)
    """
    print("\n" + "=" * 60)
    print("DATA AUGMENTATION (Sensor Noise Simulation)")
    print("=" * 60)
    
    n_samples = X_train.shape[0]
    
    # Identify feature indices by type
    temp_indices = [i for i, col in enumerate(feature_columns) 
                    if 'temp' in col.lower() and 'interaction' not in col.lower()]
    hum_indices = [i for i, col in enumerate(feature_columns) 
                   if 'hum' in col.lower() and 'interaction' not in col.lower()]
    press_indices = [i for i, col in enumerate(feature_columns) 
                     if 'press' in col.lower() and 'interaction' not in col.lower()]
    
    print(f"   Temperature features ({len(temp_indices)}): "
          f"{[feature_columns[i] for i in temp_indices]}")
    print(f"   Humidity features ({len(hum_indices)}): "
          f"{[feature_columns[i] for i in hum_indices]}")
    print(f"   Pressure features ({len(press_indices)}): "
          f"{[feature_columns[i] for i in press_indices]}")
    
    # Generate augmented copies
    augmented_X = [X_train]
    augmented_y = [y_train]
    
    for _ in range(Config.AUGMENTATION_FACTOR):
        noisy_X = X_train.copy()
        
        # Add noise to temperature features
        for idx in temp_indices:
            std = X_train[:, idx].std() if X_train[:, idx].std() > 0 else 1
            noise = np.random.normal(0, Config.NOISE_TEMP_STD / std * 0.5, n_samples)
            noisy_X[:, idx] += noise
        
        # Add noise to humidity features
        for idx in hum_indices:
            std = X_train[:, idx].std() if X_train[:, idx].std() > 0 else 1
            noise = np.random.normal(0, Config.NOISE_HUM_STD / std * 0.5, n_samples)
            noisy_X[:, idx] += noise
        
        # Add noise to pressure features (lower magnitude - LPS25 is more accurate)
        for idx in press_indices:
            std = X_train[:, idx].std() if X_train[:, idx].std() > 0 else 1
            noise = np.random.normal(0, Config.NOISE_PRESSURE_STD / std * 0.5, n_samples)
            noisy_X[:, idx] += noise
        
        augmented_X.append(noisy_X)
        augmented_y.append(y_train)
    
    X_augmented = np.vstack(augmented_X)
    y_augmented = np.vstack(augmented_y)
    
    print(f"\n[SUCCESS] Augmented: {n_samples} -> {X_augmented.shape[0]} samples "
          f"(x{Config.AUGMENTATION_FACTOR + 1})")
    
    return X_augmented, y_augmented


# ==============================================================================
# STEP 5: NEURAL NETWORK
# ==============================================================================

class NeuralNetwork:
    """
    Feedforward neural network optimized for microcontroller deployment.
    
    Architecture: MLP with sigmoid activations and softmax output.
    Training: Mini-batch SGD with momentum.
    """
    
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        Initialize network with Xavier weight initialization.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer dimensions
            output_size: Number of output classes
        """
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.n_layers = len(self.layer_sizes)
        
        # Xavier initialization
        self.weights = []
        self.biases = []
        
        for i in range(self.n_layers - 1):
            scale = np.sqrt(2.0 / (self.layer_sizes[i] + self.layer_sizes[i+1]))
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * scale
            b = np.zeros(self.layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)
        
        # Momentum buffers
        self.weight_velocities = [np.zeros_like(w) for w in self.weights]
        self.bias_velocities = [np.zeros_like(b) for b in self.biases]
        
        self.learning_rate = Config.LEARNING_RATE
        self.momentum = Config.MOMENTUM
    
    def sigmoid(self, x):
        """Sigmoid activation function with numerical stability."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid for backpropagation."""
        return x * (1 - x)
    
    def softmax(self, x):
        """Softmax activation for output layer."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, X):
        """
        Forward pass through the network.
        
        Args:
            X: Input features (batch_size, input_size)
            
        Returns:
            np.ndarray: Output probabilities (batch_size, output_size)
        """
        self.activations = [X]
        current = X
        
        # Hidden layers with sigmoid
        for i in range(len(self.weights) - 1):
            z = np.dot(current, self.weights[i]) + self.biases[i]
            current = self.sigmoid(z)
            self.activations.append(current)
        
        # Output layer with softmax
        z = np.dot(current, self.weights[-1]) + self.biases[-1]
        output = self.softmax(z)
        self.activations.append(output)
        
        return output
    
    def backward(self, y_true):
        """
        Backward pass with SGD + momentum updates.
        
        Args:
            y_true: One-hot encoded ground truth labels
        """
        m = y_true.shape[0]
        
        # Output layer gradient (cross-entropy + softmax)
        delta = self.activations[-1] - y_true
        deltas = [delta]
        
        # Hidden layer gradients
        for i in range(len(self.weights) - 2, -1, -1):
            error = np.dot(deltas[0], self.weights[i+1].T)
            delta = error * self.sigmoid_derivative(self.activations[i+1])
            deltas.insert(0, delta)
        
        # Update weights with momentum
        for i in range(len(self.weights)):
            grad_w = np.dot(self.activations[i].T, deltas[i]) / m
            grad_b = np.mean(deltas[i], axis=0)
            
            self.weight_velocities[i] = (
                self.momentum * self.weight_velocities[i] - 
                self.learning_rate * grad_w
            )
            self.bias_velocities[i] = (
                self.momentum * self.bias_velocities[i] - 
                self.learning_rate * grad_b
            )
            
            self.weights[i] += self.weight_velocities[i]
            self.biases[i] += self.bias_velocities[i]
    
    def train(self, X_train, y_train, X_val, y_val, epochs=500):
        """
        Train the network with mini-batch SGD.
        
        Args:
            X_train: Training features
            y_train: Training labels (one-hot)
            X_val: Validation features
            y_val: Validation labels (one-hot)
            epochs: Maximum training epochs
            
        Returns:
            dict: Training history (loss and accuracy curves)
        """
        print("\n" + "=" * 60)
        print("STEP 5: NEURAL NETWORK TRAINING")
        print("=" * 60)
        print(f"\nArchitecture: {self.layer_sizes}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Momentum: {self.momentum}")
        print(f"Epochs: {epochs}")
        
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        best_val_acc = 0
        best_weights = None
        best_biases = None
        patience = 50
        patience_counter = 0
        
        batch_size = Config.BATCH_SIZE
        n_batches = max(1, len(X_train) // batch_size)
        
        print("\nTraining in progress...\n")
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch training
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, len(X_train))
                self.forward(X_shuffled[start:end])
                self.backward(y_shuffled[start:end])
            
            # Evaluate
            train_loss, train_acc = self.evaluate(X_train, y_train)
            val_loss, val_acc = self.evaluate(X_val, y_val)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Progress logging
            if epoch % 50 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:4d}: Train Loss={train_loss:.4f}, "
                      f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
            
            if patience_counter >= patience:
                print(f"\n[EARLY STOPPING] No improvement for {patience} epochs")
                break
        
        # Restore best weights
        if best_weights:
            self.weights = best_weights
            self.biases = best_biases
        
        print(f"\n[SUCCESS] Best validation accuracy: {best_val_acc:.4f}")
        
        return history
    
    def evaluate(self, X, y):
        """
        Compute loss and accuracy on a dataset.
        
        Args:
            X: Features
            y: One-hot labels
            
        Returns:
            tuple: (cross_entropy_loss, accuracy)
        """
        predictions = self.forward(X)
        loss = -np.mean(np.sum(y * np.log(predictions + 1e-10), axis=1))
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y, axis=1)
        accuracy = np.mean(pred_classes == true_classes)
        return loss, accuracy
    
    def predict(self, X):
        """Return predicted class indices."""
        probas = self.forward(X)
        return np.argmax(probas, axis=1)
    
    def predict_proba(self, X):
        """Return class probabilities."""
        return self.forward(X)


# ==============================================================================
# STEP 6: MODEL EVALUATION
# ==============================================================================

def evaluate_model(nn, data):
    """
    Comprehensive model evaluation with confusion matrix and per-class metrics.
    
    Args:
        nn: Trained neural network
        data: Dictionary containing test data
        
    Returns:
        dict: Evaluation metrics
    """
    print("\n" + "=" * 60)
    print("STEP 6: MODEL EVALUATION")
    print("=" * 60)
    
    y_pred = nn.predict(data['X_test'])
    y_true = data['y_test_labels']
    
    accuracy = np.mean(y_pred == y_true)
    print(f"\n[RESULT] Overall accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    # Confusion matrix
    print("\n[CONFUSION MATRIX]")
    print("              Predicted")
    print("              LOW    MEDIUM  HIGH")
    
    confusion = np.zeros((3, 3), dtype=int)
    for true, pred in zip(y_true, y_pred):
        confusion[true, pred] += 1
    
    labels_names = ['LOW   ', 'MEDIUM', 'HIGH  ']
    for i, label in enumerate(labels_names):
        print(f"   Actual {label}: {confusion[i]}")
    
    # Per-class metrics
    print("\n[PER-CLASS METRICS]")
    
    for i, name in enumerate(['LOW', 'MEDIUM', 'HIGH']):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"   {name}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    
    # High-risk error analysis
    print(f"\n[ERROR ANALYSIS] HIGH RISK class:")
    fp_high = confusion[0, 2] + confusion[1, 2]
    fn_high = confusion[2, 0] + confusion[2, 1]
    print(f"   False positives (unnecessary alerts): {fp_high}")
    print(f"   False negatives (missed alerts): {fn_high}")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': confusion,
        'y_pred': y_pred,
        'y_true': y_true
    }


# ==============================================================================
# STEP 7: ARDUINO EXPORT
# ==============================================================================

def export_to_arduino(nn, data, filename='mildiou_nn_weights.h'):
    """
    Export neural network weights to C++ header file for Arduino.
    
    Generates a self-contained header with:
    - Network architecture constants
    - Normalization parameters
    - Weight matrices (stored in PROGMEM)
    
    Args:
        nn: Trained neural network
        data: Dictionary with feature info and normalization params
        filename: Output header file path
    """
    print("\n" + "=" * 60)
    print("STEP 7: ARDUINO EXPORT")
    print("=" * 60)
    
    with open(filename, 'w') as f:
        f.write("/*\n")
        f.write(" * " + "=" * 60 + "\n")
        f.write(" * MILDIOU NEURAL NETWORK WEIGHTS\n")
        f.write(" * " + "=" * 60 + "\n")
        f.write(f" * Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f" * Architecture: {nn.layer_sizes}\n")
        f.write(" * \n")
        f.write(" * Input Features (with pressure-based predictors):\n")
        for i, name in enumerate(data['feature_columns']):
            f.write(f" *   {i+1:2d}. {name}\n")
        f.write(" */\n\n")
        
        f.write("#ifndef MILDIOU_NN_WEIGHTS_H\n")
        f.write("#define MILDIOU_NN_WEIGHTS_H\n\n")
        
        f.write("#include <Arduino.h>\n\n")
        
        # Architecture constants
        f.write("// ============== ARCHITECTURE ==============\n")
        layers_str = ', '.join(map(str, nn.layer_sizes))
        f.write(f"const unsigned int NN_LAYERS[] = {{{layers_str}}};\n")
        f.write(f"const unsigned int NN_NUM_LAYERS = {len(nn.layer_sizes)};\n")
        f.write(f"const unsigned int NN_INPUT_SIZE = {nn.layer_sizes[0]};\n")
        f.write(f"const unsigned int NN_OUTPUT_SIZE = {nn.layer_sizes[-1]};\n\n")
        
        # Normalization parameters
        f.write("// ============== NORMALIZATION ==============\n")
        f.write("// Formula: input_norm = (input - mean) / std\n\n")
        
        means_str = ',\n    '.join([f"{m:.6f}f" for m in data['feature_means']])
        f.write(f"const float FEATURE_MEANS[{len(data['feature_means'])}] = {{\n    {means_str}\n}};\n\n")
        
        stds_str = ',\n    '.join([f"{s:.6f}f" for s in data['feature_stds']])
        f.write(f"const float FEATURE_STDS[{len(data['feature_stds'])}] = {{\n    {stds_str}\n}};\n\n")
        
        # Feature names
        f.write("// ============== FEATURE NAMES ==============\n")
        f.write("const char* FEATURE_NAMES[] = {\n")
        for name in data['feature_columns']:
            f.write(f'    "{name}",\n')
        f.write("};\n\n")
        
        # Network weights
        f.write("// ============== NETWORK WEIGHTS ==============\n\n")
        
        for layer_idx, (weights, biases) in enumerate(zip(nn.weights, nn.biases)):
            rows, cols = weights.shape
            f.write(f"// Layer {layer_idx + 1}: {rows} inputs -> {cols} outputs\n")
            
            # Weights
            f.write(f"const float PROGMEM WEIGHTS_L{layer_idx + 1}[{weights.size}] = {{\n")
            flat = weights.flatten()
            for i in range(0, len(flat), 6):
                chunk = flat[i:i+6]
                line = "    " + ", ".join([f"{w:10.6f}f" for w in chunk])
                if i + 6 < len(flat):
                    line += ","
                f.write(line + "\n")
            f.write("};\n\n")
            
            # Biases
            bias_str = ", ".join([f"{b:10.6f}f" for b in biases])
            f.write(f"const float PROGMEM BIAS_L{layer_idx + 1}[{len(biases)}] = {{\n    {bias_str}\n}};\n\n")
        
        f.write("#endif // MILDIOU_NN_WEIGHTS_H\n")
    
    print(f"\n[SUCCESS] Exported: {filename}")
    
    total_params = sum(w.size + b.size for w, b in zip(nn.weights, nn.biases))
    print(f"   Total parameters: {total_params}")
    print(f"   Flash memory: ~{total_params * 4} bytes")


def export_to_arduino_int8(nn, data, filename='mildiou_nn_weights_int8.h'):
    """
    Export quantized int8 weights for memory-constrained devices.
    
    Reduces memory footprint by 4x with ~1% accuracy loss.
    Experimental feature for future ESP32/ATtiny optimization.
    
    Args:
        nn: Trained neural network
        data: Dictionary with feature info
        filename: Output header file path
    """
    print("\n" + "=" * 60)
    print("QUANTIZED INT8 EXPORT (Experimental)")
    print("=" * 60)
    
    with open(filename, 'w') as f:
        f.write("/*\n")
        f.write(" * " + "=" * 60 + "\n")
        f.write(" * QUANTIZED INT8 NEURAL NETWORK\n")
        f.write(" * " + "=" * 60 + "\n")
        f.write(f" * Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f" * Architecture: {nn.layer_sizes}\n")
        f.write(" * \n")
        f.write(" * NOTE: Experimental feature.\n")
        f.write(" * Define USE_INT8_WEIGHTS in sketch to enable.\n")
        f.write(" */\n\n")
        
        f.write("#ifndef MILDIOU_NN_WEIGHTS_INT8_H\n")
        f.write("#define MILDIOU_NN_WEIGHTS_INT8_H\n\n")
        
        f.write("#include <Arduino.h>\n\n")
        
        # Quantize each layer
        for layer_idx, (weights, biases) in enumerate(zip(nn.weights, nn.biases)):
            w_max = np.max(np.abs(weights))
            b_max = np.max(np.abs(biases))
            
            w_scale = 127.0 / w_max if w_max > 0 else 1.0
            b_scale = 127.0 / b_max if b_max > 0 else 1.0
            
            w_int8 = np.round(weights * w_scale).astype(np.int8)
            b_int8 = np.round(biases * b_scale).astype(np.int8)
            
            f.write(f"// Layer {layer_idx + 1}\n")
            f.write(f"const float SCALE_W{layer_idx + 1} = {1.0/w_scale:.8f}f;\n")
            f.write(f"const float SCALE_B{layer_idx + 1} = {1.0/b_scale:.8f}f;\n")
            
            # Int8 weights
            flat = w_int8.flatten()
            f.write(f"const int8_t PROGMEM WEIGHTS_L{layer_idx + 1}_INT8[{len(flat)}] = {{\n")
            for i in range(0, len(flat), 12):
                chunk = flat[i:i+12]
                line = "    " + ", ".join([f"{w:4d}" for w in chunk])
                if i + 12 < len(flat):
                    line += ","
                f.write(line + "\n")
            f.write("};\n\n")
            
            # Int8 biases
            f.write(f"const int8_t PROGMEM BIAS_L{layer_idx + 1}_INT8[{len(b_int8)}] = {{\n    ")
            f.write(", ".join([f"{b:4d}" for b in b_int8]))
            f.write("\n};\n\n")
        
        f.write("#endif // MILDIOU_NN_WEIGHTS_INT8_H\n")
    
    print(f"\n[SUCCESS] Exported: {filename}")
    print(f"   Memory reduction: 4x compared to float32")
    print(f"   NOTE: Requires dequantization at inference time")


# ==============================================================================
# STEP 8: VISUALIZATION
# ==============================================================================

def create_visualizations(df, data, metrics, history):
    """
    Generate diagnostic plots for model analysis.
    
    Plots include:
    - Temperature and humidity time series
    - Atmospheric pressure with risk thresholds
    - Training/validation curves
    - Risk accumulation visualization
    - Confusion matrix heatmap
    - Predictions vs ground truth
    
    Args:
        df: Full dataset with features
        data: Training/test data dictionary
        metrics: Evaluation results
        history: Training history
    """
    print("\n" + "=" * 60)
    print("STEP 8: VISUALIZATION")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    # Plot 1: Temperature and Humidity
    ax1 = axes[0, 0]
    ax1.plot(test_df['meantemp'], 'r-', alpha=0.7, label='Temperature (C)')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(test_df['humidity'], 'b-', alpha=0.7, label='Humidity (%)')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Temperature (C)', color='red')
    ax1_twin.set_ylabel('Humidity (%)', color='blue')
    ax1.set_title('Temperature and Humidity (Test Set)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Atmospheric Pressure
    ax2 = axes[0, 1]
    ax2.plot(test_df['meanpressure'], 'g-', linewidth=1.5, label='Pressure')
    ax2.axhline(y=Config.PRESSURE_NORMAL, color='blue', linestyle='--', 
                alpha=0.7, label=f'Normal ({Config.PRESSURE_NORMAL})')
    ax2.axhline(y=Config.PRESSURE_LOW, color='red', linestyle='--', 
                alpha=0.7, label=f'Low ({Config.PRESSURE_LOW})')
    ax2.fill_between(range(len(test_df)), Config.PRESSURE_LOW, 
                     test_df['meanpressure'].values,
                     where=test_df['meanpressure'].values < Config.PRESSURE_LOW,
                     alpha=0.3, color='red', label='Risk Zone')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Pressure (hPa)')
    ax2.set_title('Atmospheric Pressure (Test Set)')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Learning Curves
    ax3 = axes[0, 2]
    ax3.plot(history['train_loss'], label='Train Loss', alpha=0.7)
    ax3.plot(history['val_loss'], label='Validation Loss', alpha=0.7)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Learning Curves')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Risk Accumulation
    ax4 = axes[1, 0]
    ax4.fill_between(range(len(test_df)), 0, test_df['risk_accumulator'].values,
                     alpha=0.7, color='orange')
    ax4.axhline(y=0.4, color='yellow', linestyle='--', alpha=0.7, label='Medium Threshold')
    ax4.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='High Threshold')
    ax4.set_xlabel('Day')
    ax4.set_ylabel('Accumulated Risk')
    ax4.set_title('Risk Accumulation (Test Set)')
    ax4.set_ylim(0, 1)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Confusion Matrix
    ax5 = axes[1, 1]
    im = ax5.imshow(metrics['confusion_matrix'], cmap='Blues')
    ax5.set_xticks([0, 1, 2])
    ax5.set_yticks([0, 1, 2])
    ax5.set_xticklabels(['Low', 'Medium', 'High'])
    ax5.set_yticklabels(['Low', 'Medium', 'High'])
    ax5.set_xlabel('Predicted')
    ax5.set_ylabel('Actual')
    ax5.set_title('Confusion Matrix')
    for i in range(3):
        for j in range(3):
            val = metrics['confusion_matrix'][i, j]
            color = 'white' if val > metrics['confusion_matrix'].max()/2 else 'black'
            ax5.text(j, i, val, ha='center', va='center', 
                    fontsize=14, fontweight='bold', color=color)
    
    # Plot 6: Predictions vs Actual
    ax6 = axes[1, 2]
    x = range(len(metrics['y_true']))
    ax6.scatter(x, metrics['y_true'], c='blue', alpha=0.5, s=30, 
                label='Actual', marker='o')
    ax6.scatter(x, metrics['y_pred'], c='red', alpha=0.5, s=30, 
                label='Predicted', marker='x')
    ax6.set_xlabel('Sample')
    ax6.set_ylabel('Risk Class')
    ax6.set_yticks([0, 1, 2])
    ax6.set_yticklabels(['Low', 'Medium', 'High'])
    ax6.set_title('Predictions vs Actual')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mildiou_results_v3.png', dpi=150, bbox_inches='tight')
    print("\n[SUCCESS] Saved: mildiou_results_v3.png")
    plt.show()


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    """
    Execute the complete ML pipeline.
    
    Pipeline stages:
    1. Data loading and preprocessing
    2. Temporal feature engineering
    3. Automatic label generation
    4. Data preparation and augmentation
    5. Neural network training
    6. Model evaluation
    7. Arduino export (float32 and int8)
    8. Visualization generation
    """
    print("\n" + "=" * 70)
    print("   MILDIOU AI PREDICTION PIPELINE - VERSION 3.0")
    print("   (Pressure-Based Features + Data Augmentation)")
    print("=" * 70)
    
    # Step 1: Load data
    df = load_data()
    
    # Step 2: Feature engineering
    df = create_temporal_features(df)
    
    # Step 3: Automatic labeling
    df = create_labels(df)
    
    # Step 4: Data preparation
    data = prepare_training_data(df)
    
    # Step 4b: Data augmentation
    X_train_aug, y_train_aug = augment_data(
        data['X_train'], 
        data['y_train'],
        data['feature_columns']
    )
    
    # Step 5: Neural network training
    input_size = len(data['feature_columns'])
    nn = NeuralNetwork(
        input_size=input_size,
        hidden_sizes=Config.HIDDEN_LAYERS,
        output_size=Config.OUTPUT_SIZE
    )
    
    print(f"\n[INFO] Architecture: {nn.layer_sizes}")
    total_params = sum(w.size + b.size for w, b in zip(nn.weights, nn.biases))
    print(f"[INFO] Total parameters: {total_params}")
    
    history = nn.train(
        X_train_aug, y_train_aug,
        data['X_test'], data['y_test'],
        epochs=Config.EPOCHS
    )
    
    # Step 6: Evaluation
    metrics = evaluate_model(nn, data)
    
    # Step 7: Export
    export_to_arduino(nn, data)
    export_to_arduino_int8(nn, data)
    
    # Step 8: Visualization
    create_visualizations(df, data, metrics, history)
    
    # Save labeled dataset
    df.to_csv('donnees_labellisees_v4.csv', index=False)
    print("\n[SUCCESS] Saved: donnees_labellisees_v4.csv")
    
    # Summary
    print("\n" + "=" * 70)
    print("   PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nGenerated files:")
    print("  1. mildiou_nn_weights.h      -> Arduino header (float32)")
    print("  2. mildiou_nn_weights_int8.h -> Quantized version (int8)")
    print("  3. mildiou_results_v3.png    -> Diagnostic plots")
    print("  4. donnees_labellisees_v4.csv -> Labeled dataset")
    print(f"\nModel: {nn.layer_sizes} | Parameters: {total_params}")


if __name__ == "__main__":
    main()