import torch


class ImdbModel(torch.nn.Module):
    def __init__(self, vocabulary, embedding_dim, hidden_dim=128):
        super().__init__()

        self.embedding = torch.nn.Embedding(
            len(vocabulary),
            embedding_dim,
            padding_idx=vocabulary["<PAD>"],

        )

        self.lstm = torch.nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.linear = torch.nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        x = self.embedding(x)

        _, (h_n, _) = self.lstm(x)
        x = torch.cat((h_n[-2], h_n[-1]), dim=1)

        x = self.linear(x)
        return x
