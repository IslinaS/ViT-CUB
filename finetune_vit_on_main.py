import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import timm
import matplotlib.pyplot as plt

# --------------------------
# CONFIG
# --------------------------
main_train_dir = '/usr/xtmp/ys298/CUB_200_2011/ece590/main_dataset/train'
main_val_dir = '/usr/xtmp/ys298/CUB_200_2011/ece590/main_dataset/val'
pretrained_weights = '/home/users/ys298/ece590/Final_project/checkpoint/vit_multihead_concepts_epoch=50.pth'
num_epochs = 20
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------------
# DATA LOADERS
# --------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(main_train_dir, transform=transform)
val_dataset = datasets.ImageFolder(main_val_dir, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

num_species = len(train_dataset.classes)

# --------------------------
# MODEL SETUP
# --------------------------
vit = timm.create_model('vit_base_patch16_224', pretrained=False)
# reshape head to 200 classes
vit.head = nn.Linear(vit.head.in_features, 200)
state_dict = torch.load(pretrained_weights)

# Remove the head weights from the checkpoint
state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}

# Load backbone weights
vit.load_state_dict(state_dict, strict=False)

nn.init.xavier_uniform_(vit.head.weight)
nn.init.zeros_(vit.head.bias)
vit.to(device)

# <<< NEW: Freeze backbone (patch_embed + blocks + norm) initially
for name, param in vit.named_parameters():
    if not name.startswith('head'):
        param.requires_grad = False

# <<< NEW: Create optimizer using only trainable params
optimizer = optim.Adam(filter(lambda p: p.requires_grad, vit.parameters()), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# --------------------------
# TRAINING LOOP
# --------------------------
freeze_epochs = 5  # <<< NEW: Number of epochs to freeze backbone

train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in range(num_epochs):
    # <<< NEW: Unfreeze backbone after freeze_epochs
    if epoch == freeze_epochs:
        print(f'Unfreezing backbone at epoch {epoch+1}')
        for param in vit.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(vit.parameters(), lr=1e-5)  # <<< NEW: use smaller lr after unfreezing

    vit.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = vit(imgs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_train_loss = total_loss / len(train_loader)
    train_accuracy = correct / total
    train_losses.append(avg_train_loss)
    train_accs.append(train_accuracy)

    # Validation phase
    vit.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = vit(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = outputs.max(1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = val_correct / val_total
    val_losses.append(avg_val_loss)
    val_accs.append(val_accuracy)

    print(f'Epoch {epoch+1}/{num_epochs} '
          f'Train Loss: {avg_train_loss:.4f} Train Acc: {train_accuracy:.4f} '
          f'Val Loss: {avg_val_loss:.4f} Val Acc: {val_accuracy:.4f}')
