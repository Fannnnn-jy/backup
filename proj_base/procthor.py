import prior

dataset = prior.load_dataset("procthor-10k")

from ai2thor.controller import Controller

house = dataset["train"][3]

controller = Controller(scene=house)

from PIL import Image

Image.fromarray(controller.last_event.frame)
# 1. 获取图像对象
img = Image.fromarray(controller.last_event.frame)

# 2. 保存到当前路径
img.save("procthor_scene_3.png")

# 3. 如果你在 Colab 想直接下载到本地电脑
try:
    from google.colab import files

    files.download("procthor_scene_3.png")
except ImportError:
    print("文件已保存至当前目录。")
