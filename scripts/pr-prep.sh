#!/usr/bin/env bash
set -euo pipefail

# Files that belong to the dev-container / deployment infrastructure
# and should NOT be included in PRs to origin.
INFRA_FILES=(
    ".claude.local.md"
    "Dockerfile"
    "scripts/docker-dev.sh"
    "scripts/deploy.sh"
    "scripts/pr-prep.sh"
    "start"
    "uv.toml"
)

usage() {
    cat <<'EOF'
Usage: pr-prep.sh <branch-name> [source-branch]

Creates a clean branch off origin/main by cherry-picking only
non-infrastructure commits from the source branch.

Arguments:
  branch-name      Name for the new PR branch
  source-branch    Branch to cherry-pick from (default: current branch)

Infrastructure files excluded from PRs:
  Dockerfile, scripts/docker-dev.sh, scripts/deploy.sh,
  scripts/pr-prep.sh, start
EOF
    exit 1
}

[ -z "${1:-}" ] && usage

BRANCH="$1"
SOURCE="${2:-HEAD}"

# Ensure origin/main is up to date
git fetch origin main

# Collect commits on source that aren't on origin/main (oldest first)
mapfile -t COMMITS < <(git log --reverse --format=%H "origin/main..${SOURCE}")

if [ ${#COMMITS[@]} -eq 0 ]; then
    echo "No commits found between origin/main and ${SOURCE}."
    exit 0
fi

# Build a regex pattern from the infra files list
infra_pattern=$(printf "|%s" "${INFRA_FILES[@]}")
infra_pattern="^(${infra_pattern:1})$"

# Classify each commit as infra-only or code
code_commits=()
skipped=()

for sha in "${COMMITS[@]}"; do
    files_changed=$(git diff-tree --no-commit-id --name-only -r "$sha")
    is_infra_only=true
    while IFS= read -r file; do
        if ! echo "$file" | grep -qE "$infra_pattern"; then
            is_infra_only=false
            break
        fi
    done <<< "$files_changed"

    if [ "$is_infra_only" = true ]; then
        skipped+=("$sha")
    else
        code_commits+=("$sha")
    fi
done

echo "Commits on ${SOURCE} since origin/main: ${#COMMITS[@]}"
echo "  Code commits to cherry-pick: ${#code_commits[@]}"
echo "  Infra-only commits to skip:  ${#skipped[@]}"
echo ""

if [ ${#skipped[@]} -gt 0 ]; then
    echo "Skipping:"
    for sha in "${skipped[@]}"; do
        echo "  $(git log --format='%h %s' -1 "$sha")"
    done
    echo ""
fi

if [ ${#code_commits[@]} -eq 0 ]; then
    echo "No code commits to cherry-pick."
    exit 0
fi

echo "Cherry-picking:"
for sha in "${code_commits[@]}"; do
    echo "  $(git log --format='%h %s' -1 "$sha")"
done
echo ""

# Create the branch and cherry-pick
git checkout -b "$BRANCH" origin/main
for sha in "${code_commits[@]}"; do
    git cherry-pick "$sha"
done

echo "Branch '${BRANCH}' is ready. To submit a PR:"
echo "  git push origin ${BRANCH}"
echo "  gh pr create --base main"
