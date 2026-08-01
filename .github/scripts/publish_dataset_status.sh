#!/usr/bin/env bash
#
# Publish the rendered dataset-status dashboard to the `gh-pages` branch.
#
#   publish_dataset_status.sh <site-directory> <destination>
#
# `<destination>` is `.` for the production dashboard or `previews/<branch>` for a run
# dispatched from a branch. Deploying a subdirectory is why this pushes a branch by hand
# rather than using actions/deploy-pages, which only ever replaces the whole site -- a
# branch run would otherwise overwrite the production dashboard.
#
# Requires GITHUB_TOKEN, GITHUB_REPOSITORY and GITHUB_SHA in the environment.

set -euo pipefail

site_directory=$1
destination=$2

if [ ! -d "${site_directory}" ]; then
    echo "No site directory at ${site_directory}" >&2
    exit 1
fi

work=$(mktemp -d)
repository_url="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"

# First deploy has no `gh-pages` yet, so fall back to an orphan branch.
if ! git clone --branch gh-pages --single-branch --depth 1 "${repository_url}" "${work}" 2>/dev/null; then
    echo "No gh-pages branch yet; creating one."
    git clone --depth 1 "${repository_url}" "${work}"
    git -C "${work}" checkout --orphan gh-pages
    git -C "${work}" rm -rf --quiet . || true
fi

if [ "${destination}" = "." ]; then
    # Replace the production dashboard but keep the previews tree, which belongs to
    # branches that have nothing to do with this run.
    find "${work}" -mindepth 1 -maxdepth 1 ! -name .git ! -name previews -exec rm -rf {} +
    target="${work}"
else
    target="${work}/${destination}"
    rm -rf "${target}"
    mkdir -p "${target}"
fi

cp -R "${site_directory}/." "${target}/"

# Without this GitHub Pages runs Jekyll, which drops files and directories whose names
# begin with an underscore.
touch "${work}/.nojekyll"

git -C "${work}" config user.name "github-actions[bot]"
git -C "${work}" config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git -C "${work}" add -A

if git -C "${work}" diff --cached --quiet; then
    echo "Dashboard unchanged; nothing to publish."
    exit 0
fi

git -C "${work}" commit -m "Dataset status: ${destination} (${GITHUB_SHA:0:7})"
git -C "${work}" push origin gh-pages
