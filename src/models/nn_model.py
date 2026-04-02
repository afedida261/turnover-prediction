import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .base_model import BaseTurnoverModel

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

    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.dropout2(x)
        return self.layer3(x)

class NeuralNetTurnover(BaseTurnoverModel):
    def __init__(self, epochs=100, batch_size=32, learning_rate=0.001, random_state=42):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else None

        X_scaled = self.scaler.fit_transform(X)
        y_array = np.asarray(y, dtype=np.float32)

        # Internal holdout for early stopping.
        can_stratify = len(np.unique(y_array)) > 1 and np.min(np.bincount(y_array.astype(int))) > 1
        if len(X_scaled) >= 20 and can_stratify:
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled,
                y_array,
                test_size=0.15,
                random_state=self.random_state,
                stratify=y_array
            )
        else:
            X_train, y_train = X_scaled, y_array
            X_val, y_val = X_scaled, y_array

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(self.device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(self.device)

        input_dim = X_train.shape[1]
        self.model = TurnoverNet(input_dim).to(self.device)

        pos_count = max(float(np.sum(y_train == 1.0)), 1.0)
        neg_count = max(float(np.sum(y_train == 0.0)), 1.0)
        pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=5,
            factor=0.5
        )

        dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float('inf')
        best_state = copy.deepcopy(self.model.state_dict())
        patience = 10
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            self.model.train()
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                logits = self.model(batch_X)
                train_loss = criterion(logits, batch_y)
                train_loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val_tensor)
                val_loss = criterion(val_logits, y_val_tensor).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                break

        self.model.load_state_dict(best_state)
                
        return self

    def predict_proba(self, X):
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits)

        return probs.cpu().numpy().flatten()

    def get_feature_importance(self):
        if self.model:
            w = self.model.layer1.weight.detach().cpu().numpy()
            importances = np.mean(np.abs(w), axis=0) 
            if self.feature_names is not None:
                return dict(zip(self.feature_names, importances))
            return importances
        return {}
