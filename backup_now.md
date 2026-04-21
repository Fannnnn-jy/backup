# Junyi 备份现状快照

生成时间：2026-04-21 02:14:23 UTC

扫描根目录：`/home/ghr/fs/Junyi`

## 直接结论

- 当前目录总大小约 `32G`，文件数约 `40055`。
- 当前目录里至少有 `35` 个文件大于 `100MB`，其中 `11` 个文件大于 `500MB`。
- 顶层目录已经是一个 Git 仓库，但工作区非常脏，`git diff --stat` 显示约 `2117 files changed`，包含大规模删除和新增。
- 目录里还有 3 个内嵌 Git 仓库：`GLEAM/`、`SLAM-Former/`、`iGibson/`。
- 因为存在超大文件、训练产物、软链接和内嵌仓库，当前目录**不适合直接当成一个完整 GitHub 仓库整体 push**。

更稳妥的做法是两层备份：

1. 导出一个“可推 GitHub 的脚本快照仓库”
2. 另外保留一份整目录冷备份到外部磁盘或别的机器

## 当前 Git 状态

- 顶层仓库：是
- 当前分支：`main`
- 最近提交：
  - `0a9a237 3.28 backup`
  - `47aba27 initial commit`
- 当前 `origin`：
  - `https://***@github.com/Fannnnn-jy/all-about-actrec.git`

补充：

- 当前 `origin` URL 里仍然带有 token 痕迹，迁移完成后建议改成 SSH URL 或不带 token 的 HTTPS URL。
- `git submodule status` 失败，说明当前仓库里有 gitlink/嵌套仓库残留，但 `.gitmodules` 不完整，不适合继续把它当一个规范父仓库维护。

## contributor / 账号状态

当前身份并不一致：

- 本地全局 Git 提交身份：`geng-haoran <ghr@berkeley.edu>`
- 当前 GitHub 连接账号：`Arayavalokitesvaro`（Yuchen Huang）

如果你希望 GitHub 上的 contributor 显示成你自己的账号，而不是当前机器用户或别人的账号，需要同时满足两件事：

1. 重新登录到正确的 GitHub 账号
2. 提交时使用该 GitHub 账号**已经验证过**的邮箱地址

也就是说，单纯改 `user.name` 不够，`user.email` 也必须对应到你自己的 GitHub 账号。

## 目录体量分布

按顶层路径粗略统计：

| 路径 | 大小 | 备注 |
| --- | ---: | --- |
| `proj/` | 25G | 主要体量来源，包含大量训练产物和结果 |
| `active-gs/` | 2.8G | 含实验输出 |
| `model_weights/` | 1.3G | 模型权重，不建议进 GitHub |
| `proj_slam/` | 759M | 含调试输出 |
| `iGibson/` | 600M | 自带内嵌 `.git` |
| `.git_repo_backups/` | 478M | 本地仓库元数据备份 |
| `GLEAM/` | 392M | 自带内嵌 `.git` |
| `.git/` | 316M | 当前顶层仓库历史 |
| `SLAM-Former/` | 183M | 自带内嵌 `.git` |
| `InfiniteVGGT/` | 96M | 普通目录，可进入脚本快照 |
| `proj_base/` | 93M | 当前工作树里有大量删除痕迹 |
| `CUT3R/` | 90M | 当前由父仓库跟踪为普通目录 |

## 大文件风险

当前扫描到的部分超大文件如下：

- `proj/runs/vggt_relative_stage1/20260325_065750/model_300.pt`：约 `3.67G`
- `proj/runs/vggt_relative_stage1/20260325_065750/model_200.pt`：约 `3.67G`
- `proj/runs/vggt_relative_stage1/20260325_065750/model_100.pt`：约 `3.67G`
- `proj/runs/vggt_relative_stage1/20260324_075519/model_300.pt`：约 `3.67G`
- `proj/src/traj_sampling/results_traj_sampling/manual_interp5_replicacad_apt_0_run_000_repeat_vggt.ply`：约 `1.66G`
- `model_weights/models--depth-anything--da3mono-large/...`：约 `1.33G`

这些文件会直接阻塞普通 GitHub push，除非改用外部对象存储或 Git LFS。但这类训练产物本来也不建议放进代码备份仓库。

## 软链接

当前已发现的软链接：

- `data/proj -> /datasets/v2p/current/proj`
- `GLEAM/data_gleam -> /home/ghr/fs/Junyi/data/proj/baseline_data/tmp/data_gleam`

Git 只会记录软链接指向的路径，不会把目标数据一起保存。因此这两处都不应被当成“完整数据备份”。

## 内嵌 Git 仓库

当前检测到的 `.git` 目录：

- `/home/ghr/fs/Junyi/.git`
- `/home/ghr/fs/Junyi/GLEAM/.git`
- `/home/ghr/fs/Junyi/SLAM-Former/.git`
- `/home/ghr/fs/Junyi/iGibson/.git`

这意味着：

- 如果直接在顶层仓库里 `git add .`，这些目录不会自然变成一个干净、完整、可恢复的“单仓库备份”
- 真正想保留它们的历史，应该分别备份
- 如果只想保留当前代码快照，应该导出一个新的“扁平快照仓库”，把这些 `.git` 元数据去掉

## 推荐执行路径

### 方案 A：最稳妥

- 外部盘或新机器做整目录冷备份
- GitHub 只备份代码、脚本、配置、说明文档

### 方案 B：当前机器上先准备一个可推送的快照仓库

仓库里已经补了一个导出脚本：

- `scripts/export_backup_snapshot.sh`
- `backup_snapshot.exclude`

用途：

- 从当前目录导出一个新的快照目录
- 自动排除 `.git`、数据、权重、训练产物、日志和常见大二进制
- 在导出目录里初始化一个新的 Git 仓库
- 不会直接改你现在这个脏工作树

建议命令：

```bash
cd /home/ghr/fs/Junyi
scripts/export_backup_snapshot.sh /tmp/Junyi_snapshot_repo
```

导出完成后，切到正确 GitHub 身份，再在导出目录里执行：

```bash
cd /tmp/Junyi_snapshot_repo
git config user.name "你的 GitHub 显示名"
git config user.email "你的 GitHub 账号已验证邮箱"
git add .
git commit -m "Backup snapshot 2026-04-21"
git remote add origin git@github.com:<your-account>/<new-repo>.git
git push -u origin main
```

## 目前不建议做的事

- 不建议直接在当前 `/home/ghr/fs/Junyi` 顶层仓库里执行一次大 `git add . && git commit`
- 不建议把 `proj/runs/`、`model_weights/`、`*.pt`、`*.ply`、`*.log` 之类内容直接塞到 GitHub
- 不建议继续保留带 token 的远端 URL

## 这次快照的意义

这份文档记录的是“现在这个目录还能不能被安全地整理成备份仓库”的现实状态。

结论很明确：

- 可以整理出一个比较全面的**代码/脚本快照仓库**
- 不能把当前 `32G` 目录原样无脑推成一个正常 GitHub 仓库
- 如果目标是“完整备份整台主机上的这个目录”，还必须配合一份 Git 之外的冷备份
