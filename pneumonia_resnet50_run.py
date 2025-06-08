# =======================
#  IMPORT LIBRARIES
# =======================
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score,
    recall_score, confusion_matrix, classification_report, roc_curve,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# =======================
#  LOGGING SETUP
# =======================
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# =======================
#  GLOBAL CONFIGS / HYPERPARAMETERS
# =======================
SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
EARLY_STOP_PATIENCE = 2
OUTPUT_DIR = "results"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")

# =======================
#  SET SEED FOR REPRODUCIBILITY
# =======================
def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =======================
#  CUSTOM DATASET CLASS
# =======================
class PneumoniaDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(np.squeeze(self.labels[idx]))  # ensure label is int
        image = Image.fromarray(image.astype(np.uint8))  # convert to PIL image
        if self.transform:
            image = self.transform(image)
        return image, label

# =======================
#  LOAD DATASET FROM .npz FILE
# =======================
def load_data():
    try:
        data = np.load('pneumoniamnist.npz')
        return data['train_images'], data['train_labels'], data['val_images'], data['val_labels'], data['test_images'], data['test_labels']
    except FileNotFoundError:
        logger.error("ERROR: File 'pneumoniamnist.npz' not found.")
        exit()
    except KeyError as e:
        logger.error(f"ERROR: Missing key in dataset: {e}")
        exit()

# =======================
#  TRAINING LOOP
# =======================
def train_model(model, train_loader, val_loader, criterion, optimizer, device):
    best_val_acc = 0.0
    best_epoch = -1
    best_val_loss = float('inf')
    patience = 0  # for early stopping

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        # ---- Training pass
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / total_train
        train_acc = correct_train / total_train

        # ---- Validation pass
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_loss /= total_val
        val_acc = correct_val / total_val

        # ---- Logging
        logger.info(f"📈 Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # ---- Checkpointing best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch + 1
            try:
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                logger.info(f"Best model saved at epoch {best_epoch}")
            except Exception as e:
                logger.error(f"ERROR: Failed to save model: {e}")

            # Save best model metadata
            with open(os.path.join(OUTPUT_DIR, "best_model_info.txt"), "w") as f:
                f.write(f"Best Epoch: {best_epoch}\n")
                f.write(f"Validation Accuracy: {best_val_acc:.4f}\n")
                f.write(f"Validation Loss: {best_val_loss:.4f}\n")

            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                logger.info("--> Early stopping triggered.")
                break

    logger.info(f"Best Model Details — Epoch: {best_epoch}, Val Acc: {best_val_acc:.4f}, Val Loss: {best_val_loss:.4f}")
    return best_val_acc

# =======================
#  EVALUATE ON TEST SET
# =======================
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # probability for class 1 (Pneumonia)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # ---- Compute evaluation metrics
    try:
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError as e:
        logger.warning(f"ERROR: Metric calculation error: {e}")
        return

    logger.info(f"\n Test Metrics: Accuracy={acc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, AUC-ROC={auc:.4f}")

    # ---- Save metrics to CSV
    df_metrics = pd.DataFrame({
        'Metric': ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'AUC-ROC'],
        'Value': [acc, f1, precision, recall, auc]
    })
    df_metrics.to_csv(os.path.join(OUTPUT_DIR, "metrics_results.csv"), index=False)

    # ---- Detailed classification report
    report = classification_report(all_labels, all_preds, target_names=["Normal", "Pneumonia"], output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(OUTPUT_DIR, "classification_report.csv"))

    # ---- Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Pneumonia"], yticklabels=["Normal", "Pneumonia"])
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.svg"))
    plt.close()

    # ---- ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.svg"))
    plt.close()

    # ---- Precision-Recall curve
    prec, rec, _ = precision_recall_curve(all_labels, all_probs)
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(os.path.join(OUTPUT_DIR, "precision_recall_curve.svg"))
    plt.close()

    # ---- Histogram of prediction confidence
    plt.hist(all_probs, bins=20)
    plt.xlabel("Predicted Probability for Pneumonia")
    plt.ylabel("Frequency")
    plt.title("Confidence Histogram")
    plt.savefig(os.path.join(OUTPUT_DIR, "prediction_confidence_hist.svg"))
    plt.close()

# =======================
#  MAIN DRIVER FUNCTION
# =======================
def main():
    set_seed()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load pre-split data
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_data()

    # ---- Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # convert to 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # ---- DataLoaders
    train_loader = DataLoader(PneumoniaDataset(train_images, train_labels, transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(PneumoniaDataset(val_images, val_labels, transform), batch_size=BATCH_SIZE)
    test_loader = DataLoader(PneumoniaDataset(test_images, test_labels, transform), batch_size=BATCH_SIZE)

    # ---- Load pretrained ResNet-50
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)  # modify final layer for binary classification
    model.to(device)

    # ---- Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ---- Training
    logger.info("--> Training started...")
    best_acc = train_model(model, train_loader, val_loader, criterion, optimizer, device)
    logger.info(f"--> Training finished. Best Validation Accuracy: {best_acc:.4f}")

    # ---- Load best model and evaluate
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    evaluate_model(model, test_loader, device)

# =======================
#  RUN SCRIPT
# =======================
if __name__ == "__main__":
    main()
