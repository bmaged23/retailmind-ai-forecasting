"""Small global LSTM with normalized windows and recursive forecasts."""
from datetime import timedelta
from .features import calendar_features
import math


def train(panel, cutoff, *, context=90, max_windows_per_product=180, epochs=20, batch_size=64, seed=42, device='cpu', patience=3, validation_days=28, hidden_units=32, learning_rate=0.001, weight_decay=0.0, metadata=None, monitor=None):
    import numpy as np
    import torch
    from torch import nn
    if context < 7 or epochs < 1 or batch_size < 1 or patience < 1 or validation_days < 1:
        raise ValueError('Invalid LSTM training settings')
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    x, y, vx, vy, aux, vaux = [], [], [], [], [], []
    validation_start = cutoff - timedelta(days=validation_days - 1)
    for code, series in panel.items():
        days = sorted(d for d in series if d <= cutoff)
        for target in days[-max_windows_per_product:]:
            required = [target - timedelta(days=i) for i in range(context, 0, -1)]
            if not all(day in series for day in required):
                continue
            values = np.asarray([series[d] for d in required], dtype=np.float32)
            scale = max(float(values.mean()), 1.0)
            features = np.log1p(values / scale)
            label = math.log1p(series[target] / scale)
            extra = list(calendar_features(target)) + metadata.at(code, target - timedelta(days=1)) if metadata else []
            if target >= validation_start:
                vx.append(features); vy.append(label); vaux.append(extra)
            else:
                x.append(features); y.append(label); aux.append(extra)
    if not x or not vx:
        raise ValueError('Insufficient complete training and chronological early-stopping windows')

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_units, batch_first=True)
            self.head = nn.Linear(hidden_units + (12 if metadata else 0), 1)

        def forward(self, inputs, extra):
            output, _ = self.lstm(inputs)
            last = output[:, -1, :]
            return self.head(torch.cat((last, extra), dim=1)).squeeze(-1)

    model = Network().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    tensor_x = torch.from_numpy(np.asarray(x)).unsqueeze(-1)
    tensor_y = torch.tensor(y, dtype=torch.float32)
    tensor_aux = torch.tensor(aux, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tensor_x, tensor_y, tensor_aux), batch_size=batch_size, shuffle=True, generator=generator)
    validation_x = torch.from_numpy(np.asarray(vx)).unsqueeze(-1)
    validation_y = torch.tensor(vy, dtype=torch.float32)
    validation_aux = torch.tensor(vaux, dtype=torch.float32)
    validation_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(validation_x, validation_y, validation_aux), batch_size=batch_size)
    best_loss = float('inf')
    best_state = None
    stale = 0
    for epoch in range(epochs):
        model.train()
        for inputs, targets, extra in loader:
            if monitor is not None: monitor.check()
            inputs, targets, extra = inputs.to(device), targets.to(device), extra.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(inputs, extra), targets)
            loss.backward()
            optimizer.step()
        model.eval()
        total_loss = 0.0
        with torch.inference_mode():
            for inputs, targets, extra in validation_loader:
                inputs, targets, extra = inputs.to(device), targets.to(device), extra.to(device)
                total_loss += nn.functional.mse_loss(model(inputs, extra), targets, reduction='sum').item()
        val_loss = total_loss / len(validation_y)
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    if device == 'cuda': torch.cuda.synchronize()
    model.uses_metadata = metadata is not None
    model.best_epoch = best_epoch
    model.epochs_run = epoch + 1
    model.best_validation_loss = best_loss
    model.eval()
    return model


def predict(model, history, horizon=28, *, context=90, device='cpu', cutoff=None, metadata_values=None, monitor=None):
    import numpy as np
    import torch
    if len(history) < context or horizon < 1:
        raise ValueError('Insufficient history or invalid horizon')
    values = list(history)
    predictions = []
    with torch.inference_mode():
        for step in range(1, horizon + 1):
            if monitor is not None: monitor.check()
            window = np.asarray(values[-context:], dtype=np.float32)
            scale = max(float(window.mean()), 1.0)
            inputs = torch.from_numpy(np.log1p(window / scale)).view(1, context, 1).to(device)
            if model.uses_metadata:
                if cutoff is None or metadata_values is None:
                    raise ValueError('Enriched forecast requires cutoff and metadata')
                aux_values = list(calendar_features(cutoff + timedelta(days=step))) + list(metadata_values)
            else:
                aux_values = []
            extra = torch.tensor([aux_values], dtype=torch.float32, device=device)
            output = float(model(inputs, extra).item())
            # Bound the log prediction to avoid numerical overflow on unstable fits.
            prediction = max(0.0, math.expm1(min(output, 20.0)) * scale)
            predictions.append(prediction)
            values.append(prediction)
    return predictions
