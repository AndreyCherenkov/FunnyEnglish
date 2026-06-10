import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ==========================================================
# DEVICE
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Устройство:", device)

# ==========================================================
# DATASET
# ==========================================================

dataset = load_dataset("PhilipMay/stsb_multi_mt", name="ru")
data = dataset["train"]

sentences1 = data["sentence1"][:10000]
sentences2 = data["sentence2"][:10000]

# нормируем оценки из диапазона [0,5] в [0,1]
scores = [s / 5.0 for s in data["similarity_score"][:10000]]

train_s1, test_s1, train_s2, test_s2, train_scores, test_scores = train_test_split(
    sentences1,
    sentences2,
    scores,
    test_size=0.2,
    random_state=42
)

# ==========================================================
# TOKENIZER
# ==========================================================

MODEL_NAME = "DeepPavlov/rubert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ==========================================================
# DATASET CLASS
# ==========================================================

class STSDataset(Dataset):
    def __init__(self, s1, s2, scores, tokenizer, max_len=64):
        self.s1 = s1
        self.s2 = s2
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.s1)

    def __getitem__(self, idx):

        enc1 = self.tokenizer(
            self.s1[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        enc2 = self.tokenizer(
            self.s2[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids1": enc1["input_ids"].squeeze(0),
            "attention_mask1": enc1["attention_mask"].squeeze(0),
            "input_ids2": enc2["input_ids"].squeeze(0),
            "attention_mask2": enc2["attention_mask"].squeeze(0),
            "score": torch.tensor(self.scores[idx], dtype=torch.float32)
        }

# ==========================================================
# DATALOADERS
# ==========================================================

train_dataset = STSDataset(
    train_s1,
    train_s2,
    train_scores,
    tokenizer
)

test_dataset = STSDataset(
    test_s1,
    test_s2,
    test_scores,
    tokenizer
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8
)

# ==========================================================
# SIAMESE BERT + COSINE
# ==========================================================

class SiameseBERT(nn.Module):

    def __init__(self, model_name):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)

    def mean_pooling(self, model_output, attention_mask):

        token_embeddings = model_output.last_hidden_state

        mask = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()

        summed = torch.sum(token_embeddings * mask, dim=1)

        summed_mask = torch.clamp(mask.sum(dim=1), min=1e-9)

        return summed / summed_mask

    def encode(self, input_ids, attention_mask):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        embeddings = self.mean_pooling(
            outputs,
            attention_mask
        )

        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=1
        )

        return embeddings

    def forward(
        self,
        input_ids1,
        attention_mask1,
        input_ids2,
        attention_mask2
    ):

        emb1 = self.encode(
            input_ids1,
            attention_mask1
        )

        emb2 = self.encode(
            input_ids2,
            attention_mask2
        )

        cosine = F.cosine_similarity(
            emb1,
            emb2,
            dim=1
        )

        # [-1,1] -> [0,1]
        cosine = (cosine + 1) / 2

        return cosine

# ==========================================================
# MODEL
# ==========================================================

model = SiameseBERT(MODEL_NAME).to(device)

criterion = nn.MSELoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=2e-5,
    weight_decay=0.01
)

# ==========================================================
# TRAINING
# ==========================================================

EPOCHS = 5

train_history = []
test_history = []

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0

    for batch in train_loader:

        optimizer.zero_grad()

        preds = model(
            batch["input_ids1"].to(device),
            batch["attention_mask1"].to(device),
            batch["input_ids2"].to(device),
            batch["attention_mask2"].to(device)
        )

        loss = criterion(
            preds,
            batch["score"].to(device)
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    train_history.append(avg_train_loss)

    # =========================
    # VALIDATION
    # =========================

    model.eval()

    test_loss = 0

    with torch.no_grad():

        for batch in test_loader:

            preds = model(
                batch["input_ids1"].to(device),
                batch["attention_mask1"].to(device),
                batch["input_ids2"].to(device),
                batch["attention_mask2"].to(device)
            )

            loss = criterion(
                preds,
                batch["score"].to(device)
            )

            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)

    test_history.append(avg_test_loss)

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"| Train Loss={avg_train_loss:.4f} "
        f"| Test Loss={avg_test_loss:.4f}"
    )

# ==========================================================
# LOSS GRAPH
# ==========================================================

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, EPOCHS + 1),
    train_history,
    marker="o",
    label="Train"
)

plt.plot(
    range(1, EPOCHS + 1),
    test_history,
    marker="s",
    label="Test"
)

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Cosine Similarity Training")
plt.grid(True)
plt.legend()

plt.show()

# ==========================================================
# INFERENCE
# ==========================================================

def compare_sentences(text1, text2):

    model.eval()

    enc1 = tokenizer(
        text1,
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt"
    )

    enc2 = tokenizer(
        text2,
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt"
    )

    with torch.no_grad():

        score = model(
            enc1["input_ids"].to(device),
            enc1["attention_mask"].to(device),
            enc2["input_ids"].to(device),
            enc2["attention_mask"].to(device)
        )

    print()
    print("Текст 1:", text1)
    print("Текст 2:", text2)
    print("Косинусное сходство:", round(score.item(), 4))
    print("-" * 60)

# ==========================================================
# TESTS
# ==========================================================

compare_sentences(
    "Разработка искусственного интеллекта — захватывающее занятие.",
    "Создавать нейросети очень интересно."
)

compare_sentences(
    "Кот лениво спит на диване.",
    "Новый закон был принят парламентом."
)

compare_sentences(
    "Машина быстро ехала по трассе.",
    "Автомобиль мчался по дороге на высокой скорости."
)