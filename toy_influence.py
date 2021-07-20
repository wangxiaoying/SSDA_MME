import torch
from torch.utils.data import DataLoader
import numpy as np
import pytorch_influence_functions as ptif

from toy import ToyDataset, train, load_data

if __name__ == "__main__":
    X_source, y_source, X_target_labeled, y_target_labeled, X_target_unlabeled, y_target_unlabeled = load_data(
        "toy_data/cluster.npy")

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
    config['test_sample_num'] = False

    influences = ptif.calc_img_wise(
        config, model, train_loader, valid_loader)

    print(influences)
