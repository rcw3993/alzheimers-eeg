import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib

class RandomForestWrapper:
    """sklearn RF wrapper - channel importances for electrode study"""
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=random_state, 
            n_jobs=-1
        )
        self.is_fitted = False
    
    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True
        # Channel importances (average across 5 bands)
        self.channel_importances = self.model.feature_importances_.reshape(-1, 5).mean(1)
        return self
    
    def predict(self, X): return self.model.predict(X)
    def predict_proba(self, X): return self.model.predict_proba(X)
    
    def save(self, path):
        joblib.dump(self.model, path)
        np.save(path.with_suffix('.npy'), self.channel_importances)
    
    @classmethod
    def load(cls, path):
        model = cls()
        model.model = joblib.load(path)
        model.channel_importances = np.load(path.with_suffix('.npy'))
        model.is_fitted = True
        return model

class Simple1DCNN(nn.Module):
    """Raw windows [batch, 19ch, 1000time] → 1D conv per channel"""
    def __init__(self, n_channels=19, n_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=25, padding=12)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=25, padding=12)
        self.pool = nn.AdaptiveAvgPool1d(10)
        self.fc = nn.Linear(64 * 10, n_classes)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)

class Simple2DCNN(nn.Module):
    """STFT [batch, 19ch, 129freq, 7time] → 2D conv on spectrograms"""
    def __init__(self, n_channels=19, n_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Linear(64 * 4 * 4, n_classes)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)

class ConnectivityMLP(nn.Module):
    """PLV pairs [batch, 171] → MLP"""
    def __init__(self, n_features=171, n_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def save_model(model, path):
    """Generic save for all models"""
    if isinstance(model, RandomForestWrapper):
        model.save(path)
    elif isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), path)
    else:
        import joblib
        joblib.dump(model, path)

def load_model(model_class, path):
    """Generic load"""
    if issubclass(model_class, RandomForestWrapper):
        return model_class.load(path)
