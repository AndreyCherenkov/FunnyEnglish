import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, confusion_matrix


class NewsDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Trainer:
    def __init__(self, model, lr=1e-3):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Device:", self.device)

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def train(self, train_loader, test_loader, epochs):

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                out = self.model(x)

                loss = self.loss_fn(out, y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            acc, cm = self.evaluate(test_loader)

            print(f"Epoch {epoch + 1} | Loss: {total_loss / len(train_loader):.4f} | Acc: {acc:.4f}")

    def evaluate(self, loader):
        self.model.eval()

        preds, labels = [], []

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)

                out = self.model(x)
                pred = torch.argmax(out, dim=1)

                preds.extend(pred.cpu().numpy())
                labels.extend(y.numpy())

        acc = accuracy_score(labels, preds)
        cm = confusion_matrix(labels, preds)

        return acc, cm
