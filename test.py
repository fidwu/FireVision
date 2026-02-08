import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt


def evaluate(model, loader, device):
    y_true = []
    y_pred = []

    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    accuracy = correct / total
    return accuracy, y_true, y_pred


class EfficientNetCustom(nn.Module):
    def __init__(self, num_classes, size_inner=100, droprate=0.2):
        super(EfficientNetCustom, self).__init__()

        self.base_model = models.efficientnet_b0(weights='IMAGENET1K_V1')

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.base_model.classifier = nn.Identity()

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.inner = nn.Linear(1280, size_inner)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(droprate)  # Add dropout
        self.output = nn.Linear(size_inner, num_classes)

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.inner(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.output(x)
        return x


test_dir = "images/test"
IMG_SIZE = 224
learning_rate = 0.01
inner_size = 200
droprate = 0.4
num_epochs = 10
batch_size = 128
num_classes = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

model = EfficientNetCustom(num_classes=num_classes, size_inner=inner_size, droprate=droprate)
model.to(device)

model.load_state_dict(torch.load("smoke_fire_classifier.pth"))
test_acc, y_true, y_pred = evaluate(model, test_loader, device)

print(f"Final Test Accuracy: {test_acc:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=test_dataset.classes
))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("Precision (macro):", precision_score(y_true, y_pred, average="macro"))
print("Recall (macro):", recall_score(y_true, y_pred, average="macro"))
print("F1 (macro):", f1_score(y_true, y_pred, average="macro"))

# Plot confusion matrix
disp = ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    display_labels=test_dataset.classes,
    cmap="Blues",
    values_format="d"
)

plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=200)
plt.show()

print("Visualization saved")
