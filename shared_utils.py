import d2l
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay 


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def init_cnn(module):
    """Initialize weights for CNNs."""
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)


def bc_accuracy(y_hat, y):
    # referenced the logistic regression notebook for this
    # then pytorch-ified it
    y_hat = torch.where(y_hat >= .5, 1, 0)
    return (y_hat == y).float().mean().cpu()


def bc_model_training(model, data, optimizer, loss_fn, epochs=10, early_stopping=True, verbose=True):
    # overall loss value for each epoch:
    loss_train = []
    loss_valid = []
    accuracy_valid = []

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 25

    for epoch in range(epochs):
        if verbose and (epoch+1)%10 == 0:
            print("epoch", epoch + 1,)

        model.train()
        loss_values = [] # loss values for each batch
        for batch_X, batch_y in data.train_dataloader():
            (batch_X_1, batch_X_2) = batch_X
            batch_X_1, batch_X_2 = batch_X_1.to(device), batch_X_2.to(device)
            batch_y = batch_y.to(device)
            preds = model(batch_X_1, batch_X_2)
            # ugh this function expects float instead of int for comparison
            # otherwise the kernel CRASHES instead of giving a type error
            loss = loss_fn(preds, batch_y.unsqueeze(1).float())
            loss_values.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_train.append(np.mean(loss_values))

        model.eval()
        loss_values = []
        accuracy_values = []
        for batch_X, batch_y in data.val_dataloader():
            (batch_X_1, batch_X_2) = batch_X
            batch_X_1, batch_X_2 = batch_X_1.to(device), batch_X_2.to(device)
            batch_y = batch_y.to(device)
            with torch.no_grad():
                preds = model(batch_X_1, batch_X_2)
                #print(preds)
                loss = loss_fn(preds, batch_y.unsqueeze(1).float())
                loss_values.append(loss.item())
                accuracy_values.append(bc_accuracy(preds, batch_y).item())

        if early_stopping:     
            val_loss = np.mean(loss_values)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the best model weights here
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break
 
        loss_valid.append(np.mean(loss_values))
        accuracy_valid.append(np.mean(accuracy_values))
        if verbose and epoch%10 == 0:
            print(f"accuracy: {accuracy_valid[-1]:.8f}")

    return model, loss_train, loss_valid, accuracy_valid


def bc_confusion_matrix(data, model, device):
    ys = []
    preds = []

    for batch_X, batch_y in data.val_dataloader():
        (batch_X_1, batch_X_2) = batch_X
        batch_X_1, batch_X_2 = batch_X_1.to(device), batch_X_2.to(device)
        batch_y = batch_y.to(device)

        y_pred = model(batch_X_1, batch_X_2)
        y_pred = (y_pred >= 0.5).long()
        y_pred = y_pred.detach().cpu().numpy()

        ys.extend(batch_y.cpu().numpy().tolist())
        preds.extend(y_pred.tolist())

    matrix = confusion_matrix(ys, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Negative", "Positive"])
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, include_values=True, cmap=plt.cm.Blues, xticks_rotation="vertical")
