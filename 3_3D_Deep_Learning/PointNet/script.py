import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# 1. Generate Synthetic 3D Shape Dataset (Sphere, Cube, Cylinder)
def generate_sphere(n_points=256):
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    phi = np.arccos(np.random.uniform(-1, 1, n_points))
    r = 1.0 + np.random.normal(0, 0.05, n_points)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.stack([x, y, z], axis=1)

def generate_cube(n_points=256):
    points = []
    points_per_face = n_points // 6
    for i in range(6):
        face_pts = np.random.uniform(-1, 1, (points_per_face, 2))
        const = np.ones((points_per_face, 1)) * (1 if i % 2 == 0 else -1)
        if i < 2:
            pts = np.hstack([const, face_pts])
        elif i < 4:
            pts = np.hstack([face_pts[:, :1], const, face_pts[:, 1:]])
        else:
            pts = np.hstack([face_pts, const])
        points.append(pts)
    points = np.vstack(points)
    if len(points) < n_points:
        extra = np.random.uniform(-1, 1, (n_points - len(points), 3))
        points = np.vstack([points, extra])
    points += np.random.normal(0, 0.05, points.shape)
    return points

def generate_cylinder(n_points=256):
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    z = np.random.uniform(-1, 1, n_points)
    r = 1.0 + np.random.normal(0, 0.05, n_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1)

def create_dataset(num_samples=100, n_points=256):
    data = []
    labels = []
    for _ in range(num_samples):
        data.append(generate_sphere(n_points))
        labels.append(0)
        data.append(generate_cube(n_points))
        labels.append(1)
        data.append(generate_cylinder(n_points))
        labels.append(2)
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.int64)

X_train, y_train = create_dataset(num_samples=60, n_points=256)
X_test, y_test = create_dataset(num_samples=20, n_points=256)

X_train_t = torch.tensor(X_train).transpose(1, 2)
y_train_t = torch.tensor(y_train)
X_test_t = torch.tensor(X_test).transpose(1, 2)
y_test_t = torch.tensor(y_test)

# 2. PointNet Architecture
class PointNetClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(PointNetClassifier, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        global_feat, argmax_idx = torch.max(x, dim=2)
        x = torch.relu(self.bn4(self.fc1(global_feat)))
        x = self.fc2(x)
        return x, argmax_idx

model = PointNetClassifier(num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 15
losses = []
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs, _ = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

model.eval()
with torch.no_grad():
    test_outputs, test_argmax = model(X_test_t)
    predictions = torch.argmax(test_outputs, dim=1)
    accuracy = (predictions == y_test_t).float().mean().item()
    print(f"Model trained! Accuracy: {accuracy*100:.1f}%")

cube_idx = np.where(y_test == 1)[0][0]
cube_points = X_test[cube_idx]
single_sample = X_test_t[cube_idx:cube_idx+1]

with torch.no_grad():
    pred, argmax_idx = model(single_sample)
    pred_class = torch.argmax(pred, dim=1).item()
    critical_indices = torch.unique(argmax_idx[0]).numpy()

class_names = ["Sphere", "Cube", "Cylinder"]

fig = plt.figure(figsize=(12, 5))
ax_loss = fig.add_subplot(121)
ax_loss.plot(range(1, epochs+1), losses, 'r-o', linewidth=2)
ax_loss.set_title("PointNet Training Loss")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Loss")
ax_loss.grid(True)

ax_3d = fig.add_subplot(122, projection='3d')
ax_3d.scatter(cube_points[:, 0], cube_points[:, 1], cube_points[:, 2], c='gray', alpha=0.3, s=20, label='All Points')
critical_pts = cube_points[critical_indices]
ax_3d.scatter(critical_pts[:, 0], critical_pts[:, 1], critical_pts[:, 2], c='red', alpha=1.0, s=60, edgecolors='k', label='Critical Points')
title_str = f"Critical Points for {class_names[y_test[cube_idx]]}\n(Predicted: {class_names[pred_class]})"
ax_3d.set_title(title_str)
ax_3d.legend()
plt.tight_layout()
plt.savefig("output.png", dpi=150)
print("Saved PointNet output visualization!")
