import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from neural_netowrk_helpers import (
    plot_reg_results,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=50,
    learning_rate=0.001,
    task="regression",
):
    """Train the model"""
    print(f"\n5. Training {task} model...")

    model = model.to(device)

    # Loss and optimizer
    if task == "regression":
        criterion = nn.MSELoss()
    else:  # classification
        criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training history
    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Learning rate scheduling
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f"best_model_{task}.pth")
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )

        # Early stopping
        if patience_counter >= 15:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    model.load_state_dict(torch.load(f"best_model_{task}.pth"))

    return model, train_losses, val_losses


def evaluate_model(model, test_loader, task, scaler_y_reg=None):
    """Evaluate model on test set"""
    print(f"\n7. Evaluating {task} model on test set...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []
    test_loss = 0.0

    if task == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCELoss()

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()

            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    print(f"Test Loss: {avg_test_loss:.4f}")

    # Plot single prediction comparison
    plt.figure(figsize=(12, 5))

    if task == "regression":
        # Denormalize for plotting
        pred_denorm = scaler_y_reg.inverse_transform(
            all_predictions.reshape(-1, 1)
        ).flatten()
        target_denorm = scaler_y_reg.inverse_transform(
            all_targets.reshape(-1, 1)
        ).flatten()

        # Plot first 100 predictions
        plot_reg_results(pred_denorm[:100], target_denorm[:100])

    return all_predictions, all_targets
