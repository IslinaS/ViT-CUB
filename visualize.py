import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load pretrained ViT model

def smart_load_vit(vit, checkpoint_path, target_num_classes=200):
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # Check if head weights exist
    head_weight = state_dict.get('head.weight', None)
    head_bias = state_dict.get('head.bias', None)
    
    if head_weight is not None:
        num_classes_in_checkpoint = head_weight.shape[0]
        print(f"Checkpoint head shape: {num_classes_in_checkpoint} classes")
        
        if num_classes_in_checkpoint != target_num_classes:
            print("→ Mismatched head; removing and keeping model head initialized")
            # Remove head weights
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}
        else:
            print("→ Matching head; keeping head weights from checkpoint")
    else:
        print("→ No head in checkpoint; using model head")
    
    vit.load_state_dict(state_dict, strict=False)
    print("Model weights loaded")

vit = timm.create_model('vit_base_patch16_224', pretrained=True)
vit.head = nn.Linear(vit.head.in_features, 200)
checkpoint_path = "/home/users/ys298/ece590/Final_project/vit_finetuned_pretrained_freeze10_epoch50.pth"

smart_load_vit(vit, checkpoint_path, target_num_classes=200)
vit.eval()

# Step 2: Define image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Step 3: Load and preprocess image
image = Image.open('/usr/xtmp/ys298/CUB_200_2011/ece590/main_dataset/train/084/Red_Legged_Kittiwake_0054_795396.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0)  # shape (1, 3, 224, 224)

# Step 4: Register forward hook on attention
attention_maps = []

def hook_fn(module, input):
    x = input[0]
    B, N, _ = x.shape  # _ = C_in
    C_out = module.qkv.weight.shape[0] // 3  # extract true C_out

    qkv = module.qkv(x)  # (B, N, 3*C_out)
    qkv = qkv.reshape(B, N, 3, module.num_heads, C_out // module.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = (q @ k.transpose(-2, -1)) * module.scale
    attn = attn.softmax(dim=-1)
    attention_maps.append(attn.detach())

# Get first block attention
hook = vit.blocks[0].attn.register_forward_pre_hook(hook_fn)

# Step 5: Forward pass
with torch.no_grad():
    vit(input_tensor)

print(f"attention block: ", vit.blocks[0].attn)

# Step 6: Process attention map
attn = attention_maps[0][0]  # shape (heads, tokens, tokens)
attn_mean = attn.mean(0)     # mean over heads → (tokens, tokens)
cls_attn = attn_mean[0, 1:]  # CLS token → patch tokens

num_patches = int(np.sqrt(cls_attn.shape[0]))
attn_map = cls_attn.reshape(num_patches, num_patches).cpu().numpy()

# Step 7: Visualize attention map
plt.imshow(attn_map, cmap='viridis')
plt.axis('off')
plt.title('Attention Map')
plt.show()

# Step 8 (optional): Overlay heatmap on original image
import cv2

heatmap = cv2.resize(attn_map, (image.width, image.height))
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
heatmap = np.uint8(255 * heatmap)
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
image_np = np.array(image)

overlay = cv2.addWeighted(image_np, 0.6, heatmap_color, 0.4, 0)
im = plt.imshow(overlay, cmap='viridis')
plt.axis('off')
plt.title('Overlayed Attention Map for Kittiwake Train')
cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.set_label('Attention Weight', rotation=270, labelpad=15)
plt.show()
plt.savefig(f'attention map kittiwake train.png')
plt.close()
print(f'attention map saved')

# Step 9: Clean up hook
hook.remove()
