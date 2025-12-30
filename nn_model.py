import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from base_model import BaseTurnoverModel

class TurnoverNet(nn.Module):
    def __init__(self, input_dim):
        super(TurnoverNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)
        self.layer2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.3)
        self.layer3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.dropout2(x)
        x = self.sigmoid(self.layer3(x))
        return x

class NeuralNetTurnover(BaseTurnoverModel):
    def __init__(self, epochs=50, batch_size=32, learning_rate=0.001):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        # Preprocessing
        self.feature_names = X.columns if hasattr(X, 'columns') else None
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # Store a sample for attribution (upto 100 samples)
        # self.X_sample = X_tensor[:100].detach().clone() # Removed to save memory and skip complex attribution
        
        # Initialize model
        input_dim = X.shape[1]
        self.model = TurnoverNet(input_dim).to(self.device)
        
        # Loss and Optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training Loop
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
        return self

    def predict_proba(self, X):
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
        return outputs.cpu().numpy().flatten()

    def get_feature_importance(self):
        """
        Calculates feature importance using simple weight magnitude of the first layer.
        This is a proxy and not as accurate as Integrated Gradients, but much faster.
        """
        if self.model:
            # Use the weights of the first layer as a proxy for feature importance
            # We take the mean absolute weight for each input feature across the 64 neurons
            w = self.model.layer1.weight.detach().cpu().numpy()
            importances = np.mean(np.abs(w), axis=0) 
            if self.feature_names is not None:
                return dict(zip(self.feature_names, importances))
            return importances
        return {}
