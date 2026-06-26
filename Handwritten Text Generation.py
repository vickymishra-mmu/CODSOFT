from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import torch
except ImportError:
    print("Error: PyTorch is not installed. Please install it using: pip install torch")
    exit()

print("Loading dataset...")

ds = load_dataset("Teklia/IAM-line")
texts = ds["train"]["text"]

corpus = "\n".join(texts)

with open("iam_corpus.txt", "w", encoding="utf-8") as f:
    f.write(corpus)

print("Corpus saved!")

# Reading corpus

text = corpus

# Creating vocabulary

chars = sorted(list(set(text)))

char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

print("Unique characters:", len(chars))

# Encoding text

encoded = [char_to_idx[c] for c in text]

seq_len = 100

X = []
y = []

for i in range(len(encoded) - seq_len):
    X.append(encoded[i:i + seq_len])
    y.append(encoded[i + seq_len])

X = torch.tensor(X)
y = torch.tensor(y)

print("Input shape:", X.shape)
print("Target shape:", y.shape)


from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = TextDataset(X, y)
loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True
)

print("Dataset ready!")


class CharRNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            batch_first=True
        )
        self.fc = nn.Linear(256, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = CharRNN(len(chars)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print(model)
print("Using device:", device)



epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    print(
        f"Epoch {epoch+1}/{epochs}, "
        f"Loss: {avg_loss:.4f}"
    )

torch.save(model.state_dict(), "char_rnn.pth")
print("Model saved successfully!")

def generate_text(seed, length=300):
    model.eval()

    generated = seed

    for _ in range(length):
        seq = generated[-100:]

        encoded_seq = [
            char_to_idx.get(c, 0)
            for c in seq
        ]

        if len(encoded_seq) < 100:
            encoded_seq = (
                [0] * (100 - len(encoded_seq))
                + encoded_seq
            )

        x = torch.tensor(
            [encoded_seq],
            dtype=torch.long
        ).to(device)

        with torch.no_grad():
            output = model(x)

        probs = torch.softmax(output[0], dim=0)
        next_idx = torch.multinomial(probs, 1).item()

        next_char = idx_to_char[next_idx]
        generated += next_char

    return generated

seed = "put down"

result = generate_text(
    seed,
    length=500
)

print("\nGenerated Text:\n")
print(result)