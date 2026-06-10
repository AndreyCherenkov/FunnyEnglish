import os
import torch
import clip
import pandas as pd

from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix



DATASET_PATH = "dataset"
MAX_IMAGES = 250

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)

print("Loading CLIP...")

model, preprocess = clip.load(
    "ViT-B/32",
    device=DEVICE
)

# Текстовые описания классов

classes = [
    "cat",
    "dog"
]

text_tokens = clip.tokenize(classes).to(DEVICE)

with torch.no_grad():

    text_features = model.encode_text(text_tokens)

    text_features /= text_features.norm(
        dim=-1,
        keepdim=True
    )

print("CLIP loaded successfully")


results = []
processed = 0

for true_label in ["cat", "dog"]:

    folder = os.path.join(
        DATASET_PATH,
        true_label
    )

    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        continue

    files = os.listdir(folder)

    for file_name in tqdm(files):

        if processed >= MAX_IMAGES:
            break

        image_path = os.path.join(
            folder,
            file_name
        )

        try:

            image = Image.open(image_path).convert("RGB")

            image_tensor = preprocess(
                image
            ).unsqueeze(0).to(DEVICE)

            with torch.no_grad():

                image_features = model.encode_image(
                    image_tensor
                )

                image_features /= image_features.norm(
                    dim=-1,
                    keepdim=True
                )

                similarity = (
                    image_features
                    @ text_features.T
                )

                pred_index = similarity.argmax().item()

            predicted_label = (
                "cat"
                if pred_index == 0
                else "dog"
            )

            results.append({
                "file": file_name,
                "true_label": true_label,
                "predicted_label": predicted_label
            })

            processed += 1

        except Exception as e:

            print(
                f"Error processing {image_path}:",
                e
            )

    if processed >= MAX_IMAGES:
        break


# ============================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================

df = pd.DataFrame(results)

df.to_csv(
    "auto_labels.csv",
    index=False,
    encoding="utf-8"
)

print("\nResults saved to auto_labels.csv")



accuracy = accuracy_score(
    df["true_label"],
    df["predicted_label"]
)

cm = confusion_matrix(
    df["true_label"],
    df["predicted_label"]
)

print("\n==============================")
print("EVALUATION")
print("==============================")

print(f"Images processed: {len(df)}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

correct = (
    df["true_label"]
    ==
    df["predicted_label"]
).sum()

wrong = len(df) - correct

print(f"\nCorrect: {correct}")
print(f"Wrong: {wrong}")

print("\nDone!")