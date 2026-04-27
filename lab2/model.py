import torch


class TextCNN(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()

        self.embedding = torch.nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0,

        )

        self.convs = torch.nn.ModuleList([
            torch.nn.Conv1d(embed_dim, 128, k)
            for k in [3, 4, 5]
        ])

        self.dropout = torch.nn.Dropout(0.2)
        self.fc = torch.nn.Linear(128 * 3, num_classes)

    def forward(self, x):
        x = self.embedding(x)        # (B, L, E)
        x = x.permute(0, 2, 1)      # (B, E, L)

        conv_outs = []
        for conv in self.convs:
            c = torch.relu(conv(x))
            p = torch.max(c, dim=2)[0]
            conv_outs.append(p)

        x = torch.cat(conv_outs, dim=1)
        x = self.dropout(x)
        x = self.fc(x)

        return x