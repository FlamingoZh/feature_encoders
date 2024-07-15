import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from defined_models.MLP_autoencoder import AE
from dataloaders.dataloader_human_behavior import HumanBehaviorDataset

np.set_printoptions(precision=3, suppress=True)


def load_MLP_model(args):
    model = AE(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        )
    return model


def data_loaders(train_data, val_data, batch_size):

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True)
    return train_loader, val_loader


def load_human_behavior_data(
        root_path="/home/zhouyuchen/dumped_human_behavior_data/",
    ):
    print("loading train set...")
    train = HumanBehaviorDataset(root_path, train=True)
    print("loading val set...")
    val = HumanBehaviorDataset(root_path, train=False)
    return train, val


def load_data(args):
    train_data, val_data = load_human_behavior_data()
    train_loader, val_loader = data_loaders(train_data, val_data, args.batch_size)
    x_train_var = np.var(train_data.data_video_clips)
    return train_data, val_data, train_loader, val_loader, x_train_var


def train(train_loader, val_loader, model, loss_func, optimizer, args):

    model.train()

    losses = []
    for idx, (clip, clip_meta) in enumerate(train_loader):

        optimizer.zero_grad()

        clip = clip.to(torch.float32)
        clip = clip.to(args.device)

        decoded = model(clip)

        loss = loss_func(decoded, clip)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    train_loss = np.mean(losses)
    val_loss = evaluate(val_loader, model, loss_func, args)

    return train_loss, val_loss


def evaluate(val_loader, model, loss_func, args):

    model.eval()

    losses = []

    for idx, (clip, clip_meta) in enumerate(val_loader):

        clip = clip.to(torch.float32)
        clip = clip.to(args.device)

        with torch.no_grad():
            decoded = model(clip)
            loss = loss_func(decoded, clip)
            losses.append(loss.item())

    return np.mean(losses)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--input_size", type=int, default=1400)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--latent_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--dataset", type=str, default="HUMAN_BEHAVIOR")
    parser.add_argument("--save", action="store_true")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_data, val_data, train_loader, val_loader, x_train_var = load_data(args)
    # Load model
    model = load_MLP_model(args)

    # move model to device
    model.to(args.device)

    # Define loss function and optimizer
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

    # Train model
    best_val_loss = float("inf")
    for epoch in range(args.max_epochs): 
        train_loss, val_loss = train(train_loader, val_loader, model, loss_func, optimizer, args)
        print(f"Epoch {epoch}, train loss: {np.round(train_loss,4)}, val loss: {np.round(val_loss, 4)}")

        # Save model
        if epoch >= 1 and val_loss < best_val_loss and args.save:
            best_val_loss = val_loss
            print(f"Best val loss: {np.round(best_val_loss, 4)}, saving model...")
            torch.save(model.state_dict(), "trained_models/MLP_model.pt")


if __name__ == "__main__":
    main()
