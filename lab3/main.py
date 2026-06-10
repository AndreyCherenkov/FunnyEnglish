import torch
from torch.utils.data import DataLoader

from lab3.Imdb_data import ImdbData, process_texts, build_vocabulary, vectorize_texts
from lab3.model import ImdbModel
from lab3.trainer import ImdbDataset, ImdbTrainer

EMBED_DIM = 300

if __name__ == "__main__":
    train_data = ImdbData("aclImdb/train")
    test_data = ImdbData("aclImdb/test")
    train_processed_texts = process_texts(train_data)
    test_processed_texts = process_texts(test_data)

    texts = [word for text, label in train_processed_texts for word in text]
    vocab = build_vocabulary(texts)

    train_vectors = vectorize_texts(train_processed_texts, vocab, max_len=300)
    test_vectors = vectorize_texts(test_processed_texts, vocab, max_len=300)

    x_train = [x for x, _ in train_vectors]
    y_train = [y for _, y in train_vectors]

    x_test = [x for x, _ in test_vectors]
    y_test = [y for _, y in test_vectors]

    train_dataset = ImdbDataset(x_train, y_train)
    test_dataset = ImdbDataset(x_test, y_test)

    model = ImdbModel(
        vocabulary=vocab,
        embedding_dim=EMBED_DIM,
    )

    trainer = ImdbTrainer(model)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    print("Train starts")
    trainer.train(train_loader, epochs=5)
    accuracy = trainer.evaluate(test_loader)
