# Minimal PyTorch fine-tuning stub (classification-style; adapt to your data/task)
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class TinyTextDS(Dataset):
    def __init__(self, xs, ys):
        self.xs = xs; self.ys = ys
    def __len__(self): return len(self.xs)
    def __getitem__(self, i): return torch.tensor(self.xs[i]).float(), torch.tensor(self.ys[i]).long()

class TinyClassifier(nn.Module):
    def __init__(self, d_in=300, d_h=128, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h), nn.ReLU(),
            nn.Linear(d_h, n_classes)
        )
    def forward(self, x): return self.net(x)

def train_demo():
    X = torch.randn(256, 300); y = (X.mean(dim=1)>0).long()
    ds = TinyTextDS(X, y); dl = DataLoader(ds, batch_size=32, shuffle=True)
    model = TinyClassifier()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(5):
        for xb, yb in dl:
            opt.zero_grad(); out = model(xb); loss = loss_fn(out, yb); loss.backward(); opt.step()
        print(f"epoch {epoch}: loss={loss.item():.4f}")
    torch.save(model.state_dict(), 'configs/torch_demo.pt')

if __name__ == '__main__':
    train_demo()
