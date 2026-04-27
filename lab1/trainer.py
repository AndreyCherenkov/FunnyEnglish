import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score

from util.tokenizer import tokenize, delete_stopwords_nltk, lemmatize_tokens_nltk, stem_tokens_nltk
from util.vectorizer import build_bag_of_words


class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Trainer:
    def __init__(self, model, lr=1e-3, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print("device founded", self.device)
        self.model = model.to(self.device)
        self.model = model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
        )

    def train(self, train_loader, val_loader=None, epochs=5):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

            if val_loader:
                acc = self.evaluate(val_loader)
                print(f"Validation accuracy: {acc:.4f}")

    def evaluate(self, data_loader):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)

                outputs = self.model(X_batch)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(y_batch.numpy())

        return accuracy_score(all_labels, all_preds)

    def predict(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X)
            preds = torch.argmax(outputs, dim=1)

        return preds.cpu().numpy()

    def predict_text(self, text, model, vocab, target_names,
                     use_stopwords=False,
                     use_lemmatize=False,
                     use_stem=False):

        tokens = tokenize(text)

        if use_stopwords:
            tokens = delete_stopwords_nltk(tokens)

        if use_lemmatize:
            tokens = lemmatize_tokens_nltk(tokens)

        if use_stem:
            tokens = stem_tokens_nltk(tokens)

        vector = build_bag_of_words([tokens], vocab)  # список из 1 текста

        device = next(model.parameters()).device
        X = torch.tensor(vector, dtype=torch.float32).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(X)
            predicted_class = torch.argmax(outputs, dim=1).item()

        return target_names[predicted_class]
