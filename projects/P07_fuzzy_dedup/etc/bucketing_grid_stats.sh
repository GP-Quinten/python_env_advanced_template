#!/usr/bin/env bash
set -euo pipefail

# Pretty header
printf "%-6s %-8s %-5s %-10s %-12s %-10s %-12s %-12s\n" \
  "ngram" "num_perm" "bands" "min_block" "max_block" "n_pairs" "num_buckets" 
echo "--------------------------------------------------------------------------------------------------------------------"

# Collect, sort, and print rows
find data/R20_bucket_unlabeled_dataset__grid -name metadata.json | \
  xargs jq -r '
    [
      .params.ngram,
      .params.num_perm,
      .params.bands,
      .params.min_block_size,
      .params.max_block_size,
      .n_pairs,
      .stats.num_buckets
    ] | @tsv
  ' | sort -k3,3n | while IFS=$'\t' read -r ngram num_perm bands min_block max_block n_pairs num_buckets; do
    printf "%-6s %-8s %-5s %-10s %-12s %-10s %-12s %-12s\n" \
      "$ngram" "$num_perm" "$bands" "$min_block" "$max_block" "$n_pairs" "$num_buckets"
done
