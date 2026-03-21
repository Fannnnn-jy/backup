import glob, os, torch
from depth_anything_3.api import DepthAnything3
import cv2

device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/da3mono-large", cache_dir='/home/ghr/fs/Junyi/model_weights')
model = model.to(device=device)

# 输入 rgb 图像1: 图像路径输入
example_path = "assets"
images = sorted(glob.glob(os.path.join(example_path, "*.png")))
# prediction = model.inference(
#     images,
# )

# 输入 rgb 图像2: numpy

img = cv2.imread(images[0])
images_numpy_list = []
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img_rgb.shape)
images_numpy_list.append(img_rgb)


# --------

prediction = model.inference(images_numpy_list)
print(prediction.depth[0])

print(prediction.processed_images.shape)
print(prediction.depth.shape)  

import matplotlib.pyplot as plt

output_dir = "depth_visualizations"
os.makedirs(output_dir, exist_ok=True)
for i in range(len(prediction.depth)):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    axes[0].imshow(prediction.processed_images[i])
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(prediction.depth[i], cmap='magma')
    axes[1].set_title("Predicted Depth (DA3-Large)")
    axes[1].axis('off')
    
    plt.savefig(os.path.join(output_dir, f"compare_{i:03d}.png"))
    plt.close()