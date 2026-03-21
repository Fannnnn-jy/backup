# actrec 数据处理与baseline 训练脚本

## 1 数据处理

> 调用方案详见 `actrec-new/run_scene2glb.sh`

- 将 replicaCAD 数据集转化为统一的 .glb 格式

- 将 HSSD 数据集转化为统一的 .glb 格式

## 2 数据可视化

- 下载到 Mac 本地直接打开

- 调用 visualize_active 脚本
    - 这里提供了默认使用 .h5 缓存体素化结果的设置，会在 .glb 文件同路径下建立缓存文件

`python -m actrec.scripts.visualize_active <path_to_glb>`

## 3 训练

> 调用方案详见 `actrec-new/run_train.sh`
