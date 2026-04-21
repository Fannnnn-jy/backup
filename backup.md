# Junyi GitHub 备份说明

## 目标

这个目录建议只把以下内容备份到 GitHub：

- 脚本
- 配置
- 说明文档

以下内容不建议直接进 GitHub：

- 数据集
- 模型权重
- 实验输出
- 本地密钥和带 token 的配置

## 当前目录的建议分类

| 路径 | 当前情况 | GitHub 备份建议 |
| --- | --- | --- |
| `backup.md`、`.gitignore`、`config_codex.sh`、`procthor_rooms_le4.txt`、`procthor_training_set.txt`、`replica_download.sh` | 根目录普通文件 | 直接纳入根备份仓库 |
| `active-gs/`、`active-gs copy/`、`depth-anything-3/`、`gleam_baseline/`、`utils/`、`vggt/` | 当前可作为普通目录处理 | 可以纳入“脚本快照仓库”，但提交前确认没有产物文件混入 |
| `CUT3R/`、`GLEAM/`、`InfiniteVGGT/`、`Pi3/`、`WinT3R/`、`proj/`、`proj_310ver_slow/`、`proj_base/`、`proj_vggt_fail/` | 目录内已有独立 `.git` | 不要直接塞进父仓库当普通目录；分别推送到各自 GitHub，或后续整理成 submodule |
| `data/` | 顶层数据入口，其中 `data/proj -> /datasets/v2p/current/proj` | 整体忽略，不进 GitHub |
| `GLEAM/data_gleam` | 软链接，指向 `data/proj/baseline_data/tmp/data_gleam` | 忽略链接本身，不备份目标数据，只在文档里记录用途和目标路径 |
| `model_weights/` | 约 1.3G 的模型权重目录 | 整体忽略；如确需保存少量权重版本，单独用 Git LFS 或外部存储 |
| `.git_repo_backups/` | 已移出的子仓库元数据备份区，目录结构为 `.git_repo_backups/<timestamp>/<repo>/.git` | 整体忽略，不推送到 GitHub；仅用于本地恢复 |
| `proj/wandb/`、`proj/runs/`、`active-gs/experiments/`、`active-gs*/depth_visualizations/`、`WinT3R/output/` 等 | 日志、训练产物、可视化输出 | 统一忽略 |

## 已确认的软链接

当前目录下已发现两个软链接：

- `data/proj -> /datasets/v2p/current/proj`
- `GLEAM/data_gleam -> /home/ghr/fs/Junyi/data/proj/baseline_data/tmp/data_gleam`

Git 对软链接的处理原则是：只记录“链接指向什么路径”，不会把目标数据一起打包进仓库。

这意味着：

- 如果保留软链接入库，换一台机器后目标路径必须重新存在
- 如果目标是本地大数据或集群路径，通常更适合直接忽略软链接
- 目标路径和重建方法应写在文档里，而不是靠 Git 保存数据

## 推荐备份方案

推荐采用“两层备份”：

1. 根目录建立一个干净的“脚本/清单仓库”
2. 每个内嵌 Git 仓库单独备份到自己的 GitHub 仓库

这样做的原因：

- 不会把子仓库历史搅在一起
- 更容易排除大数据和实验输出
- 软链接和外部数据可以只保留说明，不强行入库
- 后续如果需要，可以再把这些子仓库整理成 submodule

## 根目录仓库建议跟踪什么

根目录仓库只建议跟踪：

- 根目录脚本和说明文件
- 不带独立 `.git` 的脚本目录
- 用于恢复环境结构的文档

不建议跟踪：

- `data/`
- `model_weights/`
- 各种日志、缓存、训练输出
- 内嵌仓库里的 `.git` 元数据

## 子仓库如何处理

对已经带 `.git` 的目录，建议分别进入目录设置远端并单独推送。

示例流程：

```bash
cd /home/ghr/fs/Junyi/proj
git remote -v
git status
git remote set-url origin git@github.com:<your-name>/proj-backup.git
git push -u origin HEAD
```

其他独立仓库同理：

- `CUT3R/`
- `GLEAM/`
- `InfiniteVGGT/`
- `Pi3/`
- `WinT3R/`
- `proj_310ver_slow/`
- `proj_base/`
- `proj_vggt_fail/`

## 如果只想要一个“脚本快照仓库”

如果你的目标只是“把当前所有脚本保存一份到 GitHub”，而不是保留每个子仓库的历史，那么可以做一个单独的快照仓库。

这种方式的特点：

- 优点：最快，GitHub 上只有一个仓库
- 缺点：会丢失子仓库原始历史；软链接目标不会被一起备份

建议流程：

```bash
# 1. 复制出一个干净目录
rsync -a /home/ghr/fs/Junyi/ /path/to/Junyi_scripts_snapshot/ \
  --exclude '.git' \
  --exclude '*/.git' \
  --exclude 'data' \
  --exclude 'model_weights'

# 2. 在快照目录初始化新仓库
cd /path/to/Junyi_scripts_snapshot
git init
git add .
git commit -m "scripts snapshot"
```

这个快照仓库适合“冷备份脚本”，不适合作为长期开发主仓库。

## 当前仓库的风险提示

在直接 push 之前，建议先处理下面两个问题：

1. 当前根仓库的历史并不干净。从当前 `HEAD` 可以看到，曾经把 `id_ed25519` 作为受跟踪文件写进历史。如果这个文件是真私钥，且曾经进入任何远端或共享环境，应该视情况立即轮换。
2. 当前根仓库的 `origin` 使用了带 token 的 HTTPS URL。后续建议改成 SSH URL，或者改成不带 token 的 HTTPS URL，不要把 token 留在远端配置里。

## 提交前检查清单

每次推送前建议至少检查一次：

```bash
git status
git add -n .
git diff --cached --stat
```

如果看到下面这类内容，说明还不该提交：

- 权重文件
- 数据目录
- 训练日志
- `wandb`
- 本地密钥
- 软链接指向的大数据路径
