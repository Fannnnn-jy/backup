import glob, os, torch
from depth_anything_3.api import DepthAnything3
device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/da3mono-large", cache_dir='/home/ghr/fs/Junyi/model_weights')
model = model.to(device=device)
example_path = "assets/examples/SOH"
images = sorted(glob.glob(os.path.join(example_path, "*.png")))
prediction = model.inference(
    images,
)
# prediction.processed_images : [N, H, W, 3] uint8   array
print(prediction.processed_images.shape)
# prediction.depth            : [N, H, W]    float32 array
print(prediction.depth.shape)  
# prediction.conf             : [N, H, W]    float32 array
# print(prediction.conf.shape)  
# prediction.extrinsics       : [N, 3, 4]    float32 array # opencv w2c or colmap format
# print(prediction.extrinsics.shape)
# prediction.intrinsics       : [N, 3, 3]    float32 array
# print(prediction.intrinsics.shape)

import matplotlib.pyplot as plt

output_dir = "depth_visualizations"
os.makedirs(output_dir, exist_ok=True)
for i in range(len(prediction.depth)):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # 左边：原图
    axes[0].imshow(prediction.processed_images[i])
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # 右边：深度图
    axes[1].imshow(prediction.depth[i], cmap='magma')
    axes[1].set_title("Predicted Depth (DA3-Large)")
    axes[1].axis('off')
    
    plt.savefig(os.path.join(output_dir, f"compare_{i:03d}.png"))
    plt.close()