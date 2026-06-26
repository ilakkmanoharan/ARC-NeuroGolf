#!/usr/bin/env bash
# Configure Kaggle CLI for GitHub Actions (persists auth to GITHUB_ENV).
set -euo pipefail

if [ -n "${KAGGLE_API_TOKEN:-}" ]; then
  echo "Using KAGGLE_API_TOKEN"
  if [ -n "${GITHUB_ENV:-}" ]; then
    echo "KAGGLE_API_TOKEN=${KAGGLE_API_TOKEN}" >> "${GITHUB_ENV}"
  fi
  export KAGGLE_API_TOKEN
elif [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
  mkdir -p ~/.kaggle
  printf '{"username":"%s","key":"%s"}' "${KAGGLE_USERNAME}" "${KAGGLE_KEY}" > ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json
  echo "Using KAGGLE_USERNAME + KAGGLE_KEY (~/.kaggle/kaggle.json)"
  if [ -n "${GITHUB_ENV:-}" ]; then
    echo "KAGGLE_USERNAME=${KAGGLE_USERNAME}" >> "${GITHUB_ENV}"
    echo "KAGGLE_KEY=${KAGGLE_KEY}" >> "${GITHUB_ENV}"
  fi
else
  echo "::error::Set KAGGLE_API_TOKEN or KAGGLE_USERNAME+KAGGLE_KEY repository secrets"
  exit 1
fi
