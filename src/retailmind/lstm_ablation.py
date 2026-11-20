"""Controlled LSTM feature ablations using cutoff-safe covariates."""
from datetime import timedelta
from .features import calendar_features, demand_features

VARIANTS = ('sales', 'calendar', 'calendar_frequency', 'sales_price', 'calendar_price', 'calendar_frequency_price')


def extra_features(variant, history, target_date, metadata_values=None):
    if variant not in VARIANTS:
        raise ValueError('Unknown feature variant')
    result = []
    if variant not in ('sales', 'sales_price'):
        result.extend(calendar_features(target_date)[:4])  # weekday and month cycles
    if 'frequency' in variant:
        stats = demand_features(history)
        result.extend([stats[6], stats[7], stats[8] / 28])
    if variant.endswith('_price'):
        if metadata_values is None:
            raise ValueError('Historical price and its missingness indicator required')
        result.extend([metadata_values[0], metadata_values[3]])
    return result


def build_network(extra_dim):
    import torch
    from torch import nn
    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 32, batch_first=True)
            self.head = nn.Linear(32 + extra_dim, 1)
        def forward(self, inputs, extra):
            output, _ = self.lstm(inputs)
            return self.head(torch.cat((output[:, -1, :], extra), dim=1)).squeeze(-1)
    return Network()


def train(panel, cutoff, variant, *, metadata, context=90, max_windows=180, epochs=20,
          batch_size=64, seed=42, patience=3, device='cuda', monitor=None):
    import math
    import numpy as np
    import torch
    from torch import nn
    if variant not in VARIANTS or batch_size < 1 or context < 28:
        raise ValueError('Invalid feature variant, batch size or context')
    torch.manual_seed(seed); torch.set_num_threads(1)
    train_x, train_aux, train_y, valid_x, valid_aux, valid_y = [], [], [], [], [], []
    valid_start = cutoff - timedelta(days=27)
    products_used = set()
    for code, series in panel.items():
        days = sorted(day for day in series if day <= cutoff)
        targets = days[-max_windows:] if max_windows is not None else days
        for target in targets:
            required = [target - timedelta(days=i) for i in range(context, 0, -1)]
            if not all(day in series for day in required):
                continue
            values = np.asarray([series[d] for d in required], dtype=np.float32)
            scale = max(float(values.mean()), 1.)
            seq = np.log1p(values / scale)
            aux = extra_features(variant, values, target, metadata.at(code, target - timedelta(days=1)))
            label = math.log1p(series[target] / scale)
            if target >= valid_start:
                valid_x.append(seq); valid_aux.append(aux); valid_y.append(label)
            else:
                train_x.append(seq); train_aux.append(aux); train_y.append(label); products_used.add(code)
    if not train_x or not valid_x:
        raise ValueError('Insufficient chronological training or early-stopping windows')
    extra_dim = len(train_aux[0])
    model = build_network(extra_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    def dataset(x, aux, y):
        return torch.utils.data.TensorDataset(torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(-1),
                                              torch.tensor(aux, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    generator = torch.Generator().manual_seed(seed)
    training_count, valid_count = len(train_y), len(valid_y)
    training_data = dataset(train_x,train_aux,train_y)
    validation_data = dataset(valid_x,valid_aux,valid_y)
    del train_x, train_aux, train_y, valid_x, valid_aux, valid_y
    loader = torch.utils.data.DataLoader(training_data,batch_size=batch_size,shuffle=True,generator=generator)
    valid_loader = torch.utils.data.DataLoader(validation_data,batch_size=batch_size)
    best_loss=float('inf'); best_state=None; stale=0
    for epoch in range(epochs):
        model.train()
        for seq, aux, label in loader:
            if monitor: monitor.check()
            seq,aux,label=seq.to(device),aux.to(device),label.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.mse_loss(model(seq,aux),label);loss.backward();optimizer.step()
        model.eval(); total=0.
        with torch.inference_mode():
            for seq,aux,label in valid_loader:
                seq,aux,label=seq.to(device),aux.to(device),label.to(device)
                total+=nn.functional.mse_loss(model(seq,aux),label,reduction='sum').item()
        score=total/valid_count
        if score < best_loss-1e-6:
            best_loss=score;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_epoch=epoch+1;stale=0
        else:
            stale+=1
            if stale>=patience: break
    if best_state is None: raise ValueError('Non-finite validation loss')
    model.load_state_dict(best_state);model.eval()
    if device=='cuda': torch.cuda.synchronize()
    return model, {'best_epoch':best_epoch,'epochs_run':epoch+1,'early_stopping_loss':best_loss,
                   'training_windows':training_count,'early_stopping_windows':valid_count,'training_products':len(products_used)}


def predict(model, history, cutoff, horizon, variant, *, metadata_values, context=90, device='cuda', monitor=None):
    import math
    import numpy as np
    import torch
    if len(history)<context or horizon<1: raise ValueError('Invalid history or horizon')
    values=list(history); forecast=[]
    # Historical-frequency features remain fixed at forecast origin; model predictions
    # are continuous and must not be mistaken for observed zero/nonzero sale days.
    origin_history=list(history)
    with torch.inference_mode():
        for step in range(1,horizon+1):
            if monitor: monitor.check()
            window=np.asarray(values[-context:],dtype=np.float32);scale=max(float(window.mean()),1.)
            seq=torch.from_numpy(np.log1p(window/scale)).view(1,context,1).to(device)
            aux=extra_features(variant,origin_history,cutoff+timedelta(days=step),metadata_values)
            extra=torch.tensor([aux],dtype=torch.float32,device=device)
            output=float(model(seq,extra).item());value=max(0.,math.expm1(min(output,20.))*scale)
            forecast.append(value);values.append(value)
    return forecast
