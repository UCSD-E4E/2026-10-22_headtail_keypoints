#!/usr/bin/env bash
# Print the index of the GPU with the most free memory if it has >= $1 MiB free and is < 50% busy; exit 1 otherwise.
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits |
  awk -F', ' -v need="${1:-8000}" '$2 >= need && $3 < 50 { if ($2 > best) { best = $2; idx = $1 } } END { if (idx == "") exit 1; print idx }'
