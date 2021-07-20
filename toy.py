import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ToyModel(nn.Module):
    def __init__(self, inc=2, cls=2):
        super(ToyModel, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(inc, 3),
            nn.ReLU(),
            nn.Linear(3, cls)
        )

    def forward(self, x):
        return self.classifier(x)


def load_data(filepath):
    with open(filepath, "rb") as f:
        X_source = np.load(f)
        y_source = np.load(f)
        X_target_labeled = np.load(f)
        y_target_labeled = np.load(f)
        X_target_unlabeled = np.load(f)
        y_target_unlabeled = np.load(f)

    X_source = torch.from_numpy(X_source).type(torch.FloatTensor)
    y_source = torch.from_numpy(y_source).type(torch.LongTensor)
    X_target_labeled = torch.from_numpy(
        X_target_labeled).type(torch.FloatTensor)
    y_target_labeled = torch.from_numpy(
        y_target_labeled).type(torch.LongTensor)
    X_target_unlabeled = torch.from_numpy(
        X_target_unlabeled).type(torch.FloatTensor)
    y_target_unlabeled = torch.from_numpy(
        y_target_unlabeled).type(torch.LongTensor)

    return X_source, y_source, X_target_labeled, y_target_labeled, X_target_unlabeled, y_target_unlabeled


def load_model(filepath):
    model = ToyModel().to(device)
    model.load_state_dict(torch.load(filepath, map_location=device))
    return model


def test(model, loader, tag="Test"):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    overall_loss = 0
    is_correct = []
    with torch.no_grad():
        for data in loader:
            X, y = data
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            loss = criterion(y_pred, y)
            probs = F.softmax(y_pred, dim=-1)
            _, predvals = torch.max(probs, 1)
            is_correct.extend((predvals == y).cpu().data.numpy())
            overall_loss += loss.item() / len(loader)
    accuracy = np.mean(np.array(is_correct, dtype=np.int64))
    print(f"[{tag}] loss: {overall_loss}, accuracy: {accuracy}")
    return overall_loss, accuracy


def train(train_loader, valid_loader, test_loader, filepath="tmp.pt", seed=0, epochs=100000, log_interval=100, early_stop=10):
    torch.manual_seed(seed)

    model = ToyModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []
    best_valid_acc = 0
    best_test_acc = 0
    counter = 0
    for i in range(epochs):
        model.train()
        for data in train_loader:
            X, y = data
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            loss = criterion(y_pred, y)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i % log_interval == 0:
            print(
                f"[Train] epoch {i}, loss: {loss.item()}, last {log_interval} averaged loss: {np.mean(losses)}")
            test(model, train_loader, tag="Train")
            _, valid_acc = test(model, valid_loader, tag="Valid")
            _, test_acc = test(model, test_loader, tag="Test")
            losses = []

            if valid_acc >= best_valid_acc:
                best_valid_acc = valid_acc
                best_test_acc = test_acc
                torch.save(model.state_dict(), filepath)
            if valid_acc > best_valid_acc:
                counter = 0
            else:
                counter += 1

            if counter > early_stop:
                print("Early stop!")
                break
    print(f"Best valid acc: {best_valid_acc}, best test acc: {best_test_acc}")
    return load_model(filepath)

# Plot decision boundary


def predict(model, X):
    # Convert into numpy element to tensor
    if type(X) is np.ndarray:
        X = torch.from_numpy(X).type(torch.FloatTensor)
    # Predict and return ans
    pred = model(X)
    prob = F.softmax(pred, dim=-1)
    _, predval = torch.max(prob, 1)
    return predval.numpy()
