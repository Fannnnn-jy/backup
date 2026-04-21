# Backup Manifest

Generated at: 2026-04-21 02:17:39 UTC

Source root: `/home/ghr/fs/Junyi`
Exported root: `/tmp/Junyi_snapshot_repo_2026-04-21_v2`

## Summary

- Source size: `32G`
- Exported snapshot size: `707M`
- Exported file count: `10632`
- Source files larger than 100MB: `35`
- Source branch: `main`
- Source HEAD: `0a9a237`
- Source origin: `https://***@github.com/Fannnnn-jy/all-about-actrec.git`

## Nested Git Directories In Source

```
.
GLEAM
SLAM-Former
iGibson
```

## Symlinks In Source

```
GLEAM/data_gleam -> /home/ghr/fs/Junyi/data/proj/baseline_data/tmp/data_gleam
data/proj -> /datasets/v2p/current/proj
```

## Largest Top-Level Paths In Source

```
25G	/home/ghr/fs/Junyi/proj
2.8G	/home/ghr/fs/Junyi/active-gs
1.3G	/home/ghr/fs/Junyi/model_weights
759M	/home/ghr/fs/Junyi/proj_slam
600M	/home/ghr/fs/Junyi/iGibson
478M	/home/ghr/fs/Junyi/.git_repo_backups
392M	/home/ghr/fs/Junyi/GLEAM
316M	/home/ghr/fs/Junyi/.git
183M	/home/ghr/fs/Junyi/SLAM-Former
116M	/home/ghr/fs/Junyi/.mypy_cache
96M	/home/ghr/fs/Junyi/InfiniteVGGT
93M	/home/ghr/fs/Junyi/proj_base
```

## Snapshot Rules

This snapshot was exported with `backup_snapshot.exclude`. It is intended for
code, scripts, configs, and lightweight documentation. It intentionally skips:

- Git metadata
- dataset directories
- model weights
- training outputs
- logs
- common large binary artifacts

## Next Step

If you want the first commit to map to your own GitHub contributor identity,
configure an author email that is already verified on your GitHub account before
creating the first commit.
