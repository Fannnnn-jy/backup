#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
EXCLUDE_FILE="${SRC_ROOT}/backup_snapshot.exclude"
TARGET_ROOT="${1:-}"

usage() {
  cat <<'EOF'
Usage:
  scripts/export_backup_snapshot.sh TARGET_DIR

Environment:
  BACKUP_GIT_NAME   Optional. If set together with BACKUP_GIT_EMAIL, the script
                    will create the first commit using this author identity.
  BACKUP_GIT_EMAIL  Optional. Must be a GitHub-verified email if you want the
                    commit contributor to map to your GitHub account.
EOF
}

if [[ -z "${TARGET_ROOT}" ]]; then
  usage >&2
  exit 1
fi

if [[ ! -f "${EXCLUDE_FILE}" ]]; then
  echo "Missing exclude file: ${EXCLUDE_FILE}" >&2
  exit 1
fi

if [[ -e "${TARGET_ROOT}" ]]; then
  if find "${TARGET_ROOT}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "Target directory is not empty: ${TARGET_ROOT}" >&2
    exit 1
  fi
else
  mkdir -p "${TARGET_ROOT}"
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required but not found in PATH." >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git is required but not found in PATH." >&2
  exit 1
fi

timestamp_utc="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
date_tag="$(date -u '+%Y-%m-%d')"

echo "Exporting snapshot from ${SRC_ROOT}"
echo "Target: ${TARGET_ROOT}"

rsync -a --prune-empty-dirs --exclude-from="${EXCLUDE_FILE}" "${SRC_ROOT}/" "${TARGET_ROOT}/"

source_size="$(du -sh "${SRC_ROOT}" | awk '{print $1}')"
target_size="$(du -sh "${TARGET_ROOT}" | awk '{print $1}')"
file_count="$(find "${TARGET_ROOT}" -path '*/.git' -prune -o -type f | wc -l | tr -d ' ')"
large_file_count="$(
  find "${SRC_ROOT}" -path '*/.git' -prune -o -type f -size +100M -printf '.' \
    | wc -c \
    | tr -d ' '
)"

branch="$(git -C "${SRC_ROOT}" branch --show-current 2>/dev/null || true)"
head_commit="$(git -C "${SRC_ROOT}" rev-parse --short HEAD 2>/dev/null || true)"
origin_url="$(git -C "${SRC_ROOT}" config --get remote.origin.url 2>/dev/null || true)"
origin_url_safe="$(printf '%s' "${origin_url}" | sed 's#://[^@]*@#://***@#')"

nested_git_paths="$(
  find "${SRC_ROOT}" -maxdepth 2 -name .git -type d -printf '%h\n' \
    | sed "s#^${SRC_ROOT}/##" \
    | sed "s#^${SRC_ROOT}\$#.#" \
    | sort
)"

symlink_paths="$(
  find "${SRC_ROOT}" -maxdepth 2 -type l -printf '%P -> %l\n' | sort
)"

top_sizes="$(
  find "${SRC_ROOT}" -maxdepth 1 -mindepth 1 -printf '%f\n' \
    | sort \
    | while read -r path_name; do
        du -sh "${SRC_ROOT}/${path_name}" 2>/dev/null
      done \
    | sort -hr \
    | head -n 12
)"

cat > "${TARGET_ROOT}/BACKUP_MANIFEST.md" <<EOF
# Backup Manifest

Generated at: ${timestamp_utc}

Source root: \`${SRC_ROOT}\`
Exported root: \`${TARGET_ROOT}\`

## Summary

- Source size: \`${source_size}\`
- Exported snapshot size: \`${target_size}\`
- Exported file count: \`${file_count}\`
- Source files larger than 100MB: \`${large_file_count}\`
- Source branch: \`${branch:-unknown}\`
- Source HEAD: \`${head_commit:-unknown}\`
- Source origin: \`${origin_url_safe:-unknown}\`

## Nested Git Directories In Source

\`\`\`
${nested_git_paths:-<none>}
\`\`\`

## Symlinks In Source

\`\`\`
${symlink_paths:-<none>}
\`\`\`

## Largest Top-Level Paths In Source

\`\`\`
${top_sizes:-<none>}
\`\`\`

## Snapshot Rules

This snapshot was exported with \`backup_snapshot.exclude\`. It is intended for
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
EOF

git -C "${TARGET_ROOT}" init -b main >/dev/null

if [[ -n "${BACKUP_GIT_NAME:-}" && -n "${BACKUP_GIT_EMAIL:-}" ]]; then
  git -C "${TARGET_ROOT}" config user.name "${BACKUP_GIT_NAME}"
  git -C "${TARGET_ROOT}" config user.email "${BACKUP_GIT_EMAIL}"
  git -C "${TARGET_ROOT}" add .
  git -C "${TARGET_ROOT}" commit -m "Backup snapshot ${date_tag}" >/dev/null
  echo "Created initial commit in ${TARGET_ROOT}"
else
  echo "Initialized Git repository in ${TARGET_ROOT} without a commit."
  echo "Set BACKUP_GIT_NAME and BACKUP_GIT_EMAIL, then run:"
  echo "  git -C \"${TARGET_ROOT}\" config user.name \"Your Name\""
  echo "  git -C \"${TARGET_ROOT}\" config user.email \"your-verified-email@example.com\""
  echo "  git -C \"${TARGET_ROOT}\" add ."
  echo "  git -C \"${TARGET_ROOT}\" commit -m \"Backup snapshot ${date_tag}\""
fi
