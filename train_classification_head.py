import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from defined_models.classification_head import MLP_head
from dataloaders.dataloader_MELD import MELDLatentRepresentation

np.set_printoptions(precision=3, suppress=True)

def load_MLP_model(args):
    model = MLP_head(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        output_size=args.output_size,
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


def load_MELD_representation(
        task,
        data_path="/home/zhouyuchen/dumped_MELD/",
    ):
    print("loading train set...")
    train = MELDLatentRepresentation(data_path=data_path, split="train", task=task)
    print("loading test set...")
    val = MELDLatentRepresentation(data_path=data_path, split="test", task=task)
    return train, val


def load_data(args):
    train_data, val_data = load_MELD_representation(args.task)
    train_loader, val_loader = data_loaders(train_data, val_data, args.batch_size)
    return train_data, val_data, train_loader, val_loader


# def train(train_loader, val_loader, model, loss_func, optimizer, args):
def train(train_loader, val_loader, model, loss_func, optimizer, args, global_batch_cnt, global_batch, global_train_loss, global_val_loss):

    best_val_loss = float("inf")

    for epoch in range(args.max_epochs): 
    
        print(f"Epoch: {epoch}")
            
        model.train()

        losses = []
        for idx, (rep, label) in enumerate(train_loader):

            optimizer.zero_grad()

            rep = rep.to(torch.float32)
            rep = rep.to(args.device)
            label = label.to(args.device)

            pred_label = model(rep)

            loss = loss_func(pred_label, label)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

            val_loss = evaluate(val_loader, model, loss_func, args)
            train_loss = evaluate(train_loader, model, loss_func, args)
            print(f"Batch {idx}, train loss: {np.round(train_loss, 4)}, val loss: {np.round(val_loss, 4)}")

            global_batch_cnt += 1
            global_batch.append(global_batch_cnt)
            global_train_loss.append(train_loss)
            global_val_loss.append(val_loss)

        train_loss = np.mean(losses)
        val_loss = evaluate(val_loader, model, loss_func, args)

        # Save model
        if epoch >= 1 and val_loss < best_val_loss and args.save:
            best_val_loss = val_loss
            print(f"Best val loss: {np.round(best_val_loss, 4)}, saving model...")
            torch.save(model.state_dict(), "trained_models/MLP_classification_head.pt")


def evaluate(val_loader, model, loss_func, args):

    model.eval()

    losses = []

    for idx, (rep, label) in enumerate(val_loader):

        rep = rep.to(torch.float32)
        rep = rep.to(args.device)
        label = label.to(args.device)

        with torch.no_grad():
            pred_label = model(rep)
            loss = loss_func(pred_label, label)
            losses.append(loss.item())

    return np.mean(losses)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--input_size", type=int, default=3584)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--output_size", type=int, default=7)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--dataset", type=str, default="MELD")
    parser.add_argument("--task", type=str, default="emotion")
    parser.add_argument("--save", action="store_true")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    global_batch_cnt = 0
    global_batch = []
    global_train_loss = []
    global_val_loss = []

    # Load data
    train_data, val_data, train_loader, val_loader = load_data(args)

    # Load model
    model = load_MLP_model(args)

    # move model to device
    model.to(args.device)

    # Define loss function and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

    # Train model

    train(train_loader, val_loader, model, loss_func, optimizer, args, global_batch_cnt, global_batch, global_train_loss, global_val_loss)

    # Plot loss
    plt.plot(global_batch, global_train_loss, label="train loss")
    plt.plot(global_batch, global_val_loss, label="val loss")
    plt.legend()
    plt.xlabel("Batch")
    plt.ylabel("Cross Entropy Loss")
    plt.savefig("loss.png")

if __name__ == "__main__":
    main()
