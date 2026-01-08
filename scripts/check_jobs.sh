#!/bin/bash
# ==============================================================================
# Job monitoring utilities for HPRC cluster
# ==============================================================================

echo "=== Your Running/Pending Jobs ==="
squeue -u $USER -o "%.10i %.20j %.8T %.10M %.6D %R"

echo ""
echo "=== GPU Usage on Your Nodes ==="
for node in $(squeue -u $USER -h -o "%N" | sort -u); do
    if [ -n "$node" ]; then
        echo "Node: $node"
        ssh $node "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader" 2>/dev/null || echo "  (Unable to query GPU)"
    fi
done

echo ""
echo "=== Recent Job History ==="
sacct -u $USER --starttime=$(date -d '7 days ago' +%Y-%m-%d) \
      --format=JobID,JobName%30,State,Elapsed,MaxRSS,ExitCode \
      | head -20

echo ""
echo "=== Useful Commands ==="
echo "  squeue -u \$USER              # View your jobs"
echo "  scancel <job_id>             # Cancel a job"
echo "  scancel -u \$USER             # Cancel all your jobs"
echo "  scontrol show job <job_id>   # Detailed job info"
echo "  tail -f logs/fql_*.out       # Watch job output"
