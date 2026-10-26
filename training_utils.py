
import os
import time
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=10,
    learning_rate=0.001,
    output_dir="results",
    device=None
):
    os.makedirs(output_dir, exist_ok=True)

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    history = []
    best_val_accuracy = 0.0
    best_model_path = os.path.join(
        output_dir, "best_model.pth"
    )

    start_time = time.time()

    for epoch in range(epochs):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            train_correct += (
                predictions == labels
            ).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_accuracy = train_correct / train_total

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                predictions = outputs.argmax(dim=1)
                val_correct += (
                    predictions == labels
                ).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_accuracy = val_correct / val_total

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Accuracy: {train_accuracy:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Accuracy: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)

    training_time = time.time() - start_time

    history_df = pd.DataFrame(history)
    history_df.to_csv(
        os.path.join(output_dir, "training_history.csv"),
        index=False
    )

    plt.figure()
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(output_dir, "loss_curve.png"),
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()

    plt.figure()
    plt.plot(history_df["epoch"], history_df["train_accuracy"], label="Train Accuracy")
    plt.plot(history_df["epoch"], history_df["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(output_dir, "accuracy_curve.png"),
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()

    print(f"\nEn iyi validation accuracy: {best_val_accuracy:.4f}")
    print(f"Eğitim süresi: {training_time:.2f} saniye")
    print(f"Model kaydedildi: {best_model_path}")

    return {
        "best_val_accuracy": best_val_accuracy,
        "training_time": training_time,
        "history": history,
        "best_model_path": best_model_path
    }
