#!/usr/bin/env bash
# eval_all_models.sh — batch play evaluation for all 9 valve-turn checkpoints.
#
# Records one 30-s video per (model, p_des) combination.
# All videos collected into $REPO_DIR/eval_videos/<label>.mp4 immediately after
# each run — avoids overwrite since all saved_models/ checkpoints share the same
# video_folder (log_dir = dirname(checkpoint) in play.py).
#
# Usage (run from /home/jescobars/unitree_rl_lab, env_isaaclab must be active):
#   bash scripts/rsl_rl/eval_all_models.sh 2>&1 | tee eval_run.log
#
# Requires numpy==1.26.4 for --video.
# Check: python -c "import numpy; print(numpy.__version__)"
#
# Runtime estimate: ~3-4 min per run × 23 runs ≈ 90 min.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOGS="$REPO_DIR/logs/rsl_rl/valve_turn_g1_29dof"
PLAY="$REPO_DIR/scripts/rsl_rl/play.py"
OUT="$REPO_DIR/eval_videos"

mkdir -p "$OUT"
cd "$REPO_DIR"

run_model() {
    local label="$1"   # e.g. "v4@107PSI"
    local task="$2"
    local ckpt="$3"
    local p_des="$4"   # empty = no override (v0/v1)

    local video_dir
    video_dir="$(dirname "$ckpt")/videos/play"

    echo ""
    echo "========================================================"
    echo "  MODEL: $label   P_DES: ${p_des:-fixed}"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================================"

    # Remove any stale video from previous run in same dir
    rm -f "$video_dir"/rl-video-episode-*.mp4

    if [ -n "$p_des" ]; then
        VALVE_P_DES="$p_des" python "$PLAY" \
            --headless --num_envs 1 \
            --task "$task" \
            --checkpoint "$ckpt" \
            --video --video_length 1500
    else
        python "$PLAY" \
            --headless --num_envs 1 \
            --task "$task" \
            --checkpoint "$ckpt" \
            --video --video_length 1500
    fi

    # Move video to named output before next run overwrites it
    local src
    src="$(find "$video_dir" -name "rl-video-episode-*.mp4" | sort | tail -1)"
    if [ -n "$src" ]; then
        mv "$src" "$OUT/${label}.mp4"
        echo "  Saved: eval_videos/${label}.mp4"
    else
        echo "  WARNING: no video found in $video_dir"
    fi
}

# ── v0: fixed p_des=50, VALVE_P_DES not supported ─────────────────────────────
run_model "v0" \
    "Unitree-G1-29dof-ValveTurn-v0" \
    "$LOGS/saved_models/v0_model_100.pt" \
    ""

# ── v1: fixed p_des=50, VALVE_P_DES not supported ─────────────────────────────
run_model "v1" \
    "Unitree-G1-29dof-ValveTurn-v1" \
    "$LOGS/saved_models/turn_v1_model_300.pt" \
    ""

# ── v2 ────────────────────────────────────────────────────────────────────────
for P in 50 107 170; do
run_model "v2@${P}PSI" \
    "Unitree-G1-29dof-ValveTurn-v2" \
    "$LOGS/saved_models/turn_v2_model_500.pt" \
    "$P"
done

# ── v3 ────────────────────────────────────────────────────────────────────────
for P in 50 107 170; do
run_model "v3@${P}PSI" \
    "Unitree-G1-29dof-ValveTurn-v3" \
    "$LOGS/saved_models/turn_v3_model_600.pt" \
    "$P"
done

# ── v4 (CURRENT BEST) ─────────────────────────────────────────────────────────
for P in 50 107 170; do
run_model "v4@${P}PSI" \
    "Unitree-G1-29dof-ValveTurn-v4" \
    "$LOGS/saved_models/turn_v4_model_899.pt" \
    "$P"
done

# ── v4ae iter 800 (mid-training) ──────────────────────────────────────────────
for P in 50 107 170; do
run_model "v4ae_800@${P}PSI" \
    "Unitree-G1-29dof-ValveTurn-v4ae" \
    "$LOGS/2026-05-31_14-54-46/model_800.pt" \
    "$P"
done

# ── v4ae iter 1499 (final) ────────────────────────────────────────────────────
for P in 50 107 170; do
run_model "v4ae_1499@${P}PSI" \
    "Unitree-G1-29dof-ValveTurn-v4ae" \
    "$LOGS/2026-05-31_14-54-46/model_1499.pt" \
    "$P"
done

# ── v4a iter 1000 (mid-recovery) ──────────────────────────────────────────────
for P in 50 107 170; do
run_model "v4a_1000@${P}PSI" \
    "Unitree-G1-29dof-ValveTurn-v4a" \
    "$LOGS/2026-05-31_21-35-35/model_1000.pt" \
    "$P"
done

# ── v4a iter 1499 (final, highest SR) ─────────────────────────────────────────
for P in 50 107 170; do
run_model "v4a_1499@${P}PSI" \
    "Unitree-G1-29dof-ValveTurn-v4a" \
    "$LOGS/2026-05-31_21-35-35/model_1499.pt" \
    "$P"
done

echo ""
echo "========================================================"
echo "  ALL DONE — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Videos in: $OUT/"
ls "$OUT/"
echo "========================================================"
