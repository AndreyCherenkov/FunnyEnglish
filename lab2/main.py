import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from data import load_data, build_vocab, vectorize
from trainer import NewsDataset, Trainer
from model import TextCNN


configs = [
    (100, 5000, 50),
    (250, 15000, 150),
    (300, 25000, 300),
]


def plot_confusion_matrix(cm, classes, title):
    plt.figure()

    plt.imshow(cm, cmap="Blues")

    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.xlabel("Predicted")
    plt.ylabel("True")

    threshold = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black"
            )

    plt.tight_layout()
    plt.show()


def get_predictions(model, loader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            out = model(x)
            pred = torch.argmax(out, dim=1)

            preds.extend(pred.cpu().numpy())
            labels.extend(y.numpy())

    return np.array(labels), np.array(preds)


if __name__ == "__main__":
    train, test = load_data()

    for max_len, vocab_size, embed_dim in configs:
        print("\nCONFIG:", max_len, vocab_size, embed_dim)

        vocab = build_vocab(train.data, vocab_size)

        x_train, y_train = vectorize(train.data, train.target, vocab, max_len)
        x_test, y_test = vectorize(test.data, test.target, vocab, max_len)

        train_ds = NewsDataset(x_train, y_train)
        test_ds = NewsDataset(x_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=64)

        model = TextCNN(len(vocab), embed_dim, 20)

        trainer = Trainer(model)
        trainer.train(train_loader, test_loader, epochs=10)

        #  Матрица ошибок
        labels, preds = get_predictions(model, test_loader, trainer.device)
        cm = confusion_matrix(labels, preds)

        plot_confusion_matrix(
            cm,
            classes=train.target_names,
            title=f"Confusion Matrix (len={max_len}, vocab={vocab_size}, emb={embed_dim})"
        )

        # Свои тексты
        examples = [
            "The GPU performance of this computer is amazing",
            "Religion and faith discussions are important",
            "Space exploration and NASA missions",
            "Car engine oil and speed performance",
            "Medical treatment and disease research"
        ]

        x, _ = vectorize(examples, [0] * len(examples), vocab, max_len)
        x = torch.tensor(x, dtype=torch.long).to(trainer.device)

        model.eval()
        with torch.no_grad():
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

        print("\nPredictions for custom texts:")
        for text, pred in zip(examples, preds):
            print(f"{text} -> {train.target_names[pred]}")