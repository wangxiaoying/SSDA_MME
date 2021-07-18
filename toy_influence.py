import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
import pytorch_influence_functions as ptif

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


def train(train_loader, valid_loader, test_loader, seed=0, epochs=100000, log_interval=100, early_stop=10):
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
            test(model, valid_loader, tag="Train")
            _, valid_acc = test(model, valid_loader, tag="Valid")
            _, test_acc = test(model, test_loader, tag="Test")
            losses = []

            if valid_acc >= best_valid_acc:
                best_valid_acc = valid_acc
                best_test_acc = test_acc
            if valid_acc > best_valid_acc:
                counter = 0
            else:
                counter += 1

            if counter > early_stop:
                print("Early stop!")
                break
    print(f"Best valid acc: {best_valid_acc}, best test acc: {best_test_acc}")
    return model


if __name__ == "__main__":
    with open("toy_data/cluster.npy", "rb") as f:
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

    # Train: source
    # Valid: labeled target
    # Test: unlabeled target

    train_dataset = ToyDataset(X_source, y_source)
    valid_dataset = ToyDataset(X_target_labeled, y_target_labeled)
    test_dataset = ToyDataset(X_target_unlabeled, y_target_unlabeled)

    train_loader = DataLoader(
        train_dataset, batch_size=len(train_dataset), shuffle=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=len(valid_dataset), shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=len(test_dataset), shuffle=False)

    model = train(train_loader, valid_loader, test_loader)

    ptif.init_logging()
    config = ptif.get_default_config()
    config['gpu'] = 0 if torch.cuda.is_available() else -1
    config['dataset'] = 'Toy'
    config['num_classes'] = 2
    # make sure all labeled target are included
    config['test_sample_num'] = 100

    influences, harmful, helpful = ptif.calc_img_wise(
        config, model, train_loader, valid_loader)
