import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor


def plot_rgb_depth(rgb, depth, figsize=(8, 4), cmap="viridis"):
    """
    Visualize RGB image and depth map side by side with aligned colorbar.

    Args:
        rgb (Tensor | np.ndarray): (H, W, 3) or (3, H, W)
        depth (Tensor | np.ndarray): (H, W)
        figsize (tuple): figure size
        cmap (str): colormap for depth

    Returns:
        fig (matplotlib.figure.Figure)
    """

    # ----------- preprocess -----------
    if isinstance(rgb, Tensor):
        rgb = rgb.detach().cpu()
        if rgb.ndim == 3 and rgb.shape[0] == 3:
            rgb = rgb.permute(1, 2, 0)
        rgb = rgb.numpy()

    if isinstance(depth, Tensor):
        depth = depth.detach().cpu().numpy()

    # ----------- plotting -----------
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # RGB
    axes[0].imshow(rgb)
    axes[0].axis("off")
    axes[0].set_title("RGB")

    # Depth
    im = axes[1].imshow(depth, cmap=cmap)
    axes[1].axis("off")
    axes[1].set_title("Depth")

    # Colorbar aligned to depth
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="4%", pad=0.05)
    fig.colorbar(im, cax=cax)

    fig.tight_layout()
    return fig
