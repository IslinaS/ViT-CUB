import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_pil_image
import timm

# -------- CONFIG --------
csv_file = 'concept_labels_per_species_multilabel.csv'
num_epochs = 50
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------- LOAD LABEL MAP --------
df = pd.read_csv("concept_labels_per_species_multilabel.csv")
highlevels = df.columns[1:]
class_maps = {h: sorted(df[h].dropna().unique()) for h in highlevels}
class_to_idx = {h: {cls: i for i, cls in enumerate(classes)} for h, classes in class_maps.items()}
print(f"class to index example: ", next(iter(class_to_idx.values())))

# -------- DATASET --------
class PerHeadDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        img = read_image(img_path).float() / 255
        img = to_pil_image(img)
        if self.transform:
            img = self.transform(img)

        labels = {}
        for h in highlevels:
            val = self.df.iloc[idx][h]
            num_classes = len(class_to_idx[h])
            multi_hot = torch.zeros(num_classes)
            if not pd.isna(val) and val != 'NA':
                label_idx = class_to_idx[h][val]
                multi_hot[label_idx] = 1.0
            labels[h] = multi_hot
            # print(f"label {h}: ", labels[h])
        return img, labels


transform = Compose([
    Resize(256), CenterCrop(224),
    ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = PerHeadDataset(df, transform)
print(f"finished transforming")
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"finished loading data")

# -------- MODEL --------
vit = timm.create_model('vit_base_patch16_224', pretrained=True)
for h in highlevels:
    setattr(vit, f'head_{h}', nn.Linear(vit.head.in_features, len(class_maps[h])))

def forward_vit(x):
    features = vit.forward_features(x)  # shape [B, 197, D]
    cls_token = features[:, 0]         # shape [B, D]
    outputs = {h: getattr(vit, f'head_{h}')(cls_token) for h in highlevels}
    return outputs


vit.forward = forward_vit
vit.to(device)

# -------- OPTIMIZER & LOSSES --------
optimizer = optim.Adam(vit.parameters(), lr=3e-4)
criterions = {h: nn.BCEWithLogitsLoss() for h in highlevels} # multi-label loss

# -------- TRAINING LOOP --------
print(f"start training...")
for epoch in range(num_epochs):
    vit.train()
    total_loss = 0
    running_loss = 0
    image_count = 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device)
        labels = {h: l.to(device) for h, l in labels.items()}

        outputs = vit(imgs)
        loss = 0
        for h in highlevels:
            loss += criterions[h](outputs[h], labels[h])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = imgs.size(0)
        total_loss += loss.item()
        running_loss += loss.item()
        image_count += batch_size

        # Print every ~100 images
        if image_count >= 100:
            avg_running_loss = running_loss / (image_count / batch_size)
            print(f'  Processed {image_count} images - Batch Loss: {avg_running_loss:.4f}')
            running_loss = 0
            image_count = 0

    avg_loss = total_loss / len(loader)
    print(f'Epoch {epoch + 1}/{num_epochs} - Avg Loss: {avg_loss:.4f}')


# -------- SAVE MODEL --------
torch.save(vit.state_dict(), f'checkpoint/vit_multihead_concepts_epoch={num_epochs}.pth')
print('Model saved as vit_multihead_concepts_epoch=50.pth')
