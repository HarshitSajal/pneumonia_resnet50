# Pneumonia Detection using ResNet50

This project uses PyTorch and a pretrained ResNet-50 model to classify pneumonia from chest X-ray images using the [MedMNIST PneumoniaMNIST dataset](https://medmnist.com/).

---

## Setup

### 1. Clone this repository and move into the project folder:

bash
git clone https://github.com/HarshitSajal/pneumonia_resnet50.git
cd pneumonia-detection


### 2. Install dependencies

pip install -r requirements.txt


### 3. Download the dataset

Place the file pneumoniamnist.npz in the project root directory. You can obtain it from MedMNIST PneumoniaMNIST.

### 4. Run the script

python pneumonia_resnet50_run.py

	After training, the following will be saved in the results/ directory:

		best_model.pth: Best model weights

		best_model_info.txt: Epoch, validation accuracy, and loss

		metrics_results.csv: Accuracy, F1 score, precision, recall, AUC-ROC

		classification_report.csv: Per-class metrics

		confusion_matrix.svg: Confusion matrix visualization

		roc_curve.svg: ROC curve

		precision_recall_curve.svg: Precision-Recall curve

		prediction_confidence_hist.svg: Confidence histogram

Recommended: Python 3.11.13


