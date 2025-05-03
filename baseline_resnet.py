import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Paths
main_train_dir = '/usr/xtmp/ys298/CUB_200_2011/ece590/main_dataset/train'
main_val_dir = '/usr/xtmp/ys298/CUB_200_2011/ece590/main_dataset/val'
concept_data_dir = '/usr/xtmp/ys298/CUB_200_2011/ece590/concept_dataset'
output_dir = './resnet50_cub_output'
os.makedirs(output_dir, exist_ok=True)

# Hyperparameters
batch_size = 32
num_epochs = 50
lr = 0.001
num_workers = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Datasets and loaders
main_dataset_train = datasets.ImageFolder(main_train_dir, transform=transform)
main_dataset_val = datasets.ImageFolder(main_val_dir, transform=transform)
concept_dataset = datasets.ImageFolder(concept_data_dir, transform=transform)

main_loader_train = DataLoader(main_dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
main_loader_val = DataLoader(main_dataset_val, batch_size=batch_size, shuffle=True, num_workers=num_workers)
concept_loader = DataLoader(concept_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

print(f"finished loading datasets")

# Model
model = models.resnet50(pretrained=True)
num_classes = len(main_dataset_train.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
print(f"finished loading model")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# def train_one_epoch(loader):
#     model.train()
#     running_loss, correct, total = 0.0, 0, 0
#     for images, labels in tqdm(loader):
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * images.size(0)
#         _, preds = outputs.max(1)
#         correct += preds.eq(labels).sum().item()
#         total += labels.size(0)
#     epoch_loss = running_loss / total
#     epoch_acc = correct / total
#     return epoch_loss, epoch_acc

# def validate(loader):
#     model.eval()
#     running_loss, correct, total = 0.0, 0, 0
#     with torch.no_grad():
#         for images, labels in tqdm(loader):
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             running_loss += loss.item() * images.size(0)
#             _, preds = outputs.max(1)
#             correct += preds.eq(labels).sum().item()
#             total += labels.size(0)
#     epoch_loss = running_loss / total
#     epoch_acc = correct / total
#     return epoch_loss, epoch_acc

# # Training loop
# for epoch in range(num_epochs):
#     print(f'Epoch {epoch + 1}/{num_epochs}')
#     train_loss, train_acc = train_one_epoch(main_loader_train)
#     val_loss, val_acc = validate(main_loader_val)
#     print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
#     print(f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')
#     torch.save(model.state_dict(), os.path.join(output_dir, f'resnet50_epoch{epoch + 1}.pth'))

# test concept accuracy on the concept dataset
def run_epoch(loader, train=True):
    if train:
        model.train()
    else:
        model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Track metrics
train_losses, val_losses, concept_losses = [], [], []
train_accuracies, val_accuracies, concept_accuracies = [], [], []

# Training loop
for epoch in range(num_epochs):
    print(f'\nEpoch {epoch + 1}/{num_epochs}')
    train_loss, train_acc = run_epoch(main_loader_train, train=True)
    val_loss, val_acc = run_epoch(main_loader_val, train=False)
    concept_loss, concept_acc = run_epoch(concept_loader, train=False)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    concept_losses.append(concept_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    concept_accuracies.append(concept_acc)

    print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')
    print(f'Concept Loss: {concept_loss:.4f}, Acc: {concept_acc:.4f}')

    torch.save(model.state_dict(), os.path.join(output_dir, f'resnet50_epoch{epoch + 1}.pth'))

# Plotting
epochs = range(1, num_epochs + 1)
plt.figure()
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Val Loss')
plt.plot(epochs, concept_losses, label='Concept Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')
plt.savefig(os.path.join(output_dir, 'loss_plot.png'))

plt.figure()
plt.plot(epochs, train_accuracies, label='Train Acc')
plt.plot(epochs, val_accuracies, label='Val Acc')
plt.plot(epochs, concept_accuracies, label='Concept Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')
plt.savefig(os.path.join(output_dir, 'accuracy_plot.png'))
