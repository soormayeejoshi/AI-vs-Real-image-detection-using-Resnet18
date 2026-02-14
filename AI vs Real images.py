# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms, models
import shap
from PIL import Image
from tqdm import tqdm


# %% [markdown]
# # Load Dataset

# %%
data_dir = r"D:\AI Mini project (AI vs Real image)\archive"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=transform)
test_data  = datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=transform)

# Split training into train & validation
train_size = int(0.8 * len(train_data))
val_size   = len(train_data) - train_size
train_set, val_set = random_split(train_data, [train_size, val_size])

# %% [markdown]
# # Handle class imbalance (novelty 4)

# %%
class_counts = np.bincount([y for _, y in train_set])
weights = 1. / torch.tensor(class_counts, dtype=torch.float)
sample_weights = [weights[y] for _, y in train_set]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_set, batch_size=32, sampler=sampler)
val_loader   = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_data, batch_size=32, shuffle=False)

print(f"✅ Data Loaded: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_data)}")


# %% [markdown]
# # 2️ Model: Fine-Tuned ResNet18 (novelty 3️)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # freeze base layers

num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 2),  # real vs fake
    nn.Softmax(dim=1)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# %% [markdown]
# # 3️ Training Loop

# %%
def train_model(model, train_loader, val_loader, epochs=5):
    train_loss, val_loss = [], []
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train = running_loss / len(train_loader)
        train_loss.append(avg_train)

        # Validation
        model.eval()
        val_running = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_running += criterion(model(X), y).item()
        avg_val = val_running / len(val_loader)
        val_loss.append(avg_val)
        print(f"Epoch {epoch+1}: Train Loss={avg_train:.4f}, Val Loss={avg_val:.4f}")

    return train_loss, val_loss

train_model(model, train_loader, val_loader)

# %% [markdown]
# # 4️ Evaluation (Accuracy)

# %%
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = torch.argmax(model(X), 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100 * correct / total

acc = evaluate(model, test_loader)
print(f"✅ Test Accuracy: {acc:.2f}%")

# %% [markdown]
# 5️ Explainability: Grad-CAM (novelty 1️⃣)

# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
model.to(device)





# %%
def generate_gradcam(model, input_tensor, target_layer):
    model.eval()
    activations = {}
    gradients = {}

    # ---- forward hook ----
    def forward_hook(module, inp, out):
        activations["value"] = out
        # Register gradient hook directly on tensor output
        out.register_hook(lambda grad: gradients.setdefault("value", grad))

    # ---- register only the forward hook ----
    fwd_handle = target_layer.register_forward_hook(forward_hook)

    # forward pass
    input_tensor.requires_grad_()
    output = model(input_tensor)
    pred_class = output.argmax(dim=1)
    score = output[0, pred_class]
    model.zero_grad()

    # backward pass
    score.backward()

    # ---- compute Grad-CAM ----
    grad = gradients["value"]
    act = activations["value"]

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * act).sum(dim=1, keepdim=True))

    cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # remove hooks
    fwd_handle.remove()

    return cam, pred_class.item()



# %%
sample_img, label = next(iter(test_loader))
sample_img = sample_img[0:1].to(device)

import matplotlib.pyplot as plt
plt.imshow(np.transpose(sample_img[0].cpu().numpy(), (1, 2, 0)))
plt.title("Original Image Used for GradCAM")
plt.axis("off")
plt.show()


grad_cam, pred_class = generate_gradcam(model, sample_img, model.layer4)


# %%
img = np.transpose(sample_img[0].detach().cpu().numpy(), (1, 2, 0))  # fixed line
plt.figure(figsize=(5,5))
plt.imshow(img)
plt.imshow(grad_cam, cmap='jet', alpha=0.5)
plt.title(f"GradCAM Heatmap | Pred: {pred_class} | True: {label[0].item()}")
plt.axis('off')
plt.show()


# %%
# Get one batch
sample_batch = next(iter(test_loader))

# This contains image tensors and labels
images, labels = sample_batch

# Find the file paths using test_data.imgs
file_path = test_data.imgs[0][0]  # 0 means first image in dataset
print("🖼️ Image file used for Grad-CAM:", file_path)


# %% [markdown]
# # 7️ Robustness Testing (Novelty 3)

# %%
def test_robustness(model, loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            noisy_images = add_noise(images)
            images, labels = noisy_images.to(device), labels.to(device)

            # ⚙️ ensure inputs have grad enabled (fixes retain_grad issue)
            images.requires_grad_(True)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = 100 * correct / total
    print(f"Accuracy under noise: {acc:.2f}%")

    

# %%
# Convert sample image to numpy for visualization
img = np.transpose(sample_img[0].detach().cpu().numpy(), (1, 2, 0))

# Plot Grad-CAM output
plt.figure(figsize=(10,5))

# Original image
plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

# Grad-CAM overlay
plt.subplot(1,2,2)
plt.imshow(img)
plt.imshow(grad_cam, cmap='jet', alpha=0.5)
plt.title(f"Grad-CAM (Predicted: {pred_class})")
plt.axis("off")

plt.show()


# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_true, y_pred = [], []

# Model predictions on test data
model.eval()
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        preds = model(X).argmax(1)
        y_true.extend(y.cpu())
        y_pred.extend(preds.cpu())

# Generate and display confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Real', 'AI'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - CIFAKE Classifier")
plt.show()



