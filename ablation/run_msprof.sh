#!/bin/bash
# ============================================================================
# ASCEND-RSFA Ablation Study — msprof Profiling Script
# ============================================================================
# Runs each ablation version under msprof to collect detailed PMU data:
#   - Cube matrix multiply time
#   - Vector operation time
#   - MTE data movement time
#   - MTE/Cube/Vector overlap time
#   - Pipeline stall time
#   - Synchronization wait time
#
# Usage: bash run_msprof.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_BASE="${SCRIPT_DIR}/msprof_results"

echo "=============================================="
echo "ASCEND-RSFA Ablation Study — msprof Profiling"
echo "=============================================="
echo "Output base: ${OUTPUT_BASE}"
echo ""

# List of versions to profile
VERSIONS=(
    "01_official:Official (Baseline)"
    "02_v1_kv_advance:V1 (K-V Address Advance)"
    "03_v2_numcores:V2 (numcores Dynamic)"
    "04_v3_pipeline_optimized:V3 (Compile Config)"
    "05_ascend_rsfa:ASCEND-RSFA (Causal Split)"
)

mkdir -p "${OUTPUT_BASE}"

for entry in "${VERSIONS[@]}"; do
    mod_name="${entry%%:*}"
    label="${entry##*:}"

    echo ""
    echo "--- Profiling: ${label} ---"

    script_path="${SCRIPT_DIR}/${mod_name}.py"
    output_dir="${OUTPUT_BASE}/${mod_name}"

    if [ ! -f "${script_path}" ]; then
        echo "  ERROR: Script not found: ${script_path}"
        continue
    fi

    # Run msprof with task-based AI Core monitoring
    # --aic-mode=task-based: collect per-task AI Core metrics
    # --output: output directory for profiler data
    cmd="msprof --output=${output_dir} --aic-mode=task-based python ${script_path}"
    echo "  Command: ${cmd}"
    eval ${cmd}

    if [ $? -eq 0 ]; then
        echo "  [DONE] Results saved to: ${output_dir}"
    else
        echo "  [FAILED] msprof returned error for ${label}"
    fi
done

echo ""
echo "=============================================="
echo "All profiling complete."
echo "Results directory: ${OUTPUT_BASE}"
echo ""
echo "To view results, use:"
echo "  msprof --parse=on ${OUTPUT_BASE}/<version>"
echo "=============================================="
