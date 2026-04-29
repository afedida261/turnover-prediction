import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .base_model import BaseTurnoverModel


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.linear(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)
        return out + residual


class RegularizedMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, 128)
        self.input_bn = nn.BatchNorm1d(128)
        self.input_act = nn.LeakyReLU(negative_slope=0.1)
        self.input_dropout = nn.Dropout(0.5)

        self.residual = ResidualBlock(128, dropout=0.3)

        self.hidden = nn.Linear(128, 64)
        self.hidden_bn = nn.BatchNorm1d(64)
        self.hidden_act = nn.LeakyReLU(negative_slope=0.1)
        self.hidden_dropout = nn.Dropout(0.3)

        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.input_bn(x)
        x = self.input_act(x)
        x = self.input_dropout(x)

        x = self.residual(x)

        x = self.hidden(x)
        x = self.hidden_bn(x)
        x = self.hidden_act(x)
        x = self.hidden_dropout(x)

        return self.output(x)


class RegularizedMLPTurnover(BaseTurnoverModel):
    def __init__(self, epochs=150, batch_size=32, lr=0.001, n_folds=5, random_state=42):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_folds = n_folds
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names = None
        self.models = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _train_single_model(self, model, X_train, y_train, X_val, y_val):
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32).to(self.device),
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(self.device),
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(self.device)

        pos_count = max(float(np.sum(y_train == 1.0)), 1.0)
        neg_count = max(float(np.sum(y_train == 0.0)), 1.0)
        pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=5,
            factor=0.5,
        )

        best_val_loss = float('inf')
        best_state = copy.deepcopy(model.state_dict())
        early_stopping_patience = 15
        epochs_without_improvement = 0

        for _ in range(self.epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_tensor)
                val_loss = criterion(val_logits, y_val_tensor).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                break

        model.load_state_dict(best_state)
        return model

    def fit(self, X, y):
        self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else None

        X_np = np.asarray(X)
        y_np = np.asarray(y, dtype=np.float32)
        X_scaled = self.scaler.fit_transform(X_np)

        unique, counts = np.unique(y_np, return_counts=True)
        if len(unique) < 2:
            raise ValueError("RegularizedMLPTurnover requires at least two target classes.")

        min_class_count = int(np.min(counts))
        effective_folds = max(2, min(self.n_folds, min_class_count))
        skf = StratifiedKFold(
            n_splits=effective_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        self.models = []
        for train_idx, val_idx in skf.split(X_scaled, y_np):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_np[train_idx], y_np[val_idx]

            model = RegularizedMLP(input_dim=X_scaled.shape[1]).to(self.device)
            model = self._train_single_model(model, X_train, y_train, X_val, y_val)
            self.models.append(model)

        self.model = self.models[0] if self.models else None
        return self

    def predict_proba(self, X):
        if not self.models:
            raise ValueError("Model has not been fitted yet.")

        X_scaled = self.scaler.transform(np.asarray(X))
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        all_probs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(X_tensor)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.append(probs)

        return np.mean(np.vstack(all_probs), axis=0)

    def get_feature_importance(self):
        if not self.models:
            return {}

        fold_importances = []
        for model in self.models:
            w = model.input_layer.weight.detach().cpu().numpy()
            fold_importances.append(np.mean(np.abs(w), axis=0))

        importances = np.mean(np.vstack(fold_importances), axis=0)
        if self.feature_names is not None:
            return dict(zip(self.feature_names, importances))
        return importances
