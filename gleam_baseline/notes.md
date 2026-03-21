 我已把 GLEAM 的场景元数据处理接进现有环境：现在可以读取 eval_128 的 range_gt/voxel_size/layout/init，用它们来做边界、初始位姿和网格映射，仍保持“只用
  深度”的观测输入不变。

  细节与位置：

    scene bounds。envs/active_mapping_env.py:121
  - 增加 range+voxel 的 patch 映射函数，保持与 GLEAM 的 index 计算一致。envs/patchify.py:26
  - 新增配置项用于指向 eval_128（以及可选固定 scene）。learning/config.py:8

  用法（只需配置 env）：

  - gleam_data_dir 指向你的数据根目录，例如 /data/gleam_baseline/data/eval_128；默认会用目录名 eval_128 作为文件前缀。若前缀不一致，设置
    gleam_dataset_name 即可。envs/active_mapping_env.py:121
  - 如需固定场景，设置 gleam_scene_index>=0；否则默认随机并可在 reset 重新抽样。envs/active_mapping_env.py:131

