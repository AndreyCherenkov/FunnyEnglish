from util.data_loader import load_train_dataset_20newsgroups, load_test_dataset_20newsgroups
from util.tokenizer import process
from lab1.util import l2_normalize
from util.vectorizer import build_vocabulary, build_bag_of_words
from util.tokenizer import tokenize_texts

from model import BoWClassifier
from trainer import TextDataset, Trainer

from torch.utils.data import DataLoader

STOP_WORDS = False
LEMMATIZE = False
STEMMING = False
MAX_WORDS = 10000
EPOCHS = 10
LEARNING_RATE = 1e-3

# Загрузка данных
train_data = load_train_dataset_20newsgroups()
test_data = load_test_dataset_20newsgroups()

# Токенизация
train_tokens = process(
    tokenize_texts(train_data),
    use_stopwords=STOP_WORDS,
    use_lemmatize=LEMMATIZE,
    use_stem=STEMMING
)
test_tokens = process(
    tokenize_texts(test_data),
    use_stopwords=STOP_WORDS,
    use_lemmatize=LEMMATIZE,
    use_stem=STEMMING
)

# Словарь
vocab = build_vocabulary(train_tokens, max_word=MAX_WORDS)

# Векторизация + нормализация
X_train = build_bag_of_words(train_tokens, vocab)
X_test = build_bag_of_words(test_tokens, vocab)

X_train = l2_normalize(X_train)
X_test = l2_normalize(X_test)

y_train = train_data.target
y_test = test_data.target

# Dataset + Loader
train_dataset = TextDataset(X_train, y_train)
test_dataset = TextDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Модель
model = BoWClassifier(
    input_dim=len(vocab),
    num_classes=len(train_data.target_names)
)

trainer = Trainer(model, lr=LEARNING_RATE)

# Обучение
trainer.train(train_loader, val_loader=test_loader, epochs=EPOCHS)

# Финальная оценка
accuracy = trainer.evaluate(test_loader)
print("Final test accuracy:", accuracy)

my_text = """
Computer hardware refers to the physical components of a computer system — the tangible parts you can see and touch. 
These components work together to process data, store information, and execute programs. 
Understanding hardware is essential for anyone studying computer science,
information technology, or digital systems.

At the core of every computer is the Central Processing Unit (CPU). 
The CPU is often called the “brain” of the computer because it performs calculations, 
executes instructions, and manages the flow of information between different components. 
Modern CPUs contain multiple cores,
allowing them to perform several tasks simultaneously and improve overall performance.
"""

predicted_topic = trainer.predict_text(
    text=my_text,
    model=model,
    vocab=vocab,
    target_names=train_data.target_names,
    use_stopwords=True,
    use_lemmatize=True,
    use_stem=False
)

print("Predicted topic:", predicted_topic)
