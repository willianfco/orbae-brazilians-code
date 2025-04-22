import torch
from torch.cuda.amp import autocast
from torch import nn

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def age_train_loop(dataloader, model, criterion, mse_metric, optimizer, device, scaler):
    size = len(dataloader.dataset)
    total_loss = 0
    total_mse = 0

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        with autocast():
            pred = model(X)
            loss = criterion(pred, y.view(-1, 1))
            mse = mse_metric(pred, y.view(-1, 1))

        total_loss += loss.item()
        total_mse += mse.item()

        # Backpropagation
        optimizer.zero_grad()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            mse, current = mse.item(), (batch + 1) * len(X)
            print(
                f"batch: {batch+1} | loss: {loss:>5f} | MSE: {mse:>5f} | images: [{current:>5d}/{size:>5d}]"
            )

    avg_loss = total_loss / len(dataloader)
    avg_mse = total_mse / len(dataloader)
    return avg_loss, avg_mse


def age_valid_loop(dataloader, model, criterion, mse_metric, device):
    num_batches = len(dataloader)
    valid_loss = 0
    valid_mse = 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            with autocast():
                pred = model(X)
                valid_loss += criterion(pred, y.view(-1, 1)).item()
                valid_mse += mse_metric(pred, y.view(-1, 1)).item()

    avg_loss = valid_loss / num_batches
    avg_mse = valid_mse / num_batches
    return avg_loss, avg_mse

def accuracy(pred, y):
    _, predicted = torch.max(pred.data, 1)
    total = y.size(0)
    correct = (predicted == y).sum().item()
    return correct / total

def sex_train_loop(dataloader, model, criterion, optimizer, device, scaler):
    size = len(dataloader.dataset)
    total_loss = 0
    total_accuracy = 0

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        with autocast():
            pred = model(X)
            loss = criterion(pred, y)
            acc = accuracy(pred, y)

        total_loss += loss.item()
        total_accuracy += acc

        # Backpropagation
        optimizer.zero_grad()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            acc, current = acc, (batch + 1) * len(X)
            print(
                f"batch: {batch+1} | loss: {loss:>5f} | accuracy: {acc:>5f} | images: [{current:>5d}/{size:>5d}]"
            )

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    return avg_loss, avg_accuracy


def sex_valid_loop(dataloader, model, criterion, device):
    num_batches = len(dataloader)
    valid_loss = 0
    valid_accuracy = 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            with autocast():
                pred = model(X)
                valid_loss += criterion(pred, y).item()
                valid_accuracy += accuracy(pred, y)

    avg_loss = valid_loss / num_batches
    avg_accuracy = valid_accuracy / num_batches
    return avg_loss, avg_accuracy