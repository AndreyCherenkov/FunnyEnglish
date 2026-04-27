import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset


class ImdbDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class ImdbTrainer:
    def __init__(self, model, lr=1e-3):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        print("Device:", self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss = torch.nn.CrossEntropyLoss()

    def train(self, loader, epochs = 10):
        for epoch in range(epochs):
            self.model.train()

            total_loss = 0
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(x)
                loss = self.loss(outputs, y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")
            print(f"Validation accuracy: {self.evaluate(loader)}")

    def evaluate(self, loader):
        self.model.eval()

        predictions = []
        labels = []

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)

                outputs = self.model(x)
                predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                labels.extend(y.numpy())
        return accuracy_score(labels, predictions)