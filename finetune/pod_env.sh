# MI300X profile for the 12-hour pod session.   Usage:  source finetune/pod_env.sh
#
# 192 GB HBM lets us run a plain-bf16 LoRA on the 32B coder - no QLoRA, no
# quantization crutches. Expect ~100-140 GB in flight during training
# (weights ~65 GB + activations/optimizer); rocm-smi evidence is logged into
# run_metadata.json automatically.

export PYTHONUTF8=1

# --- model scale (the headline) ---------------------------------------------
export FT_BASE_MODEL="Qwen/Qwen2.5-Coder-32B-Instruct"
# fallback if the 32B smoke run misbehaves on ROCm (decide at the smoke gate):
#   export FT_BASE_MODEL="Qwen/Qwen2.5-Coder-14B-Instruct"
#   export FT_BASE_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"

# --- VRAM-funded capacity -----------------------------------------------------
export FT_MAX_SEQ_LEN=4096            # longer examples survive the length cap
export FT_PER_DEVICE_BATCH_SIZE=8     # real batches instead of accumulation
export FT_GRAD_ACCUM=2                # effective batch stays 16
export FT_LORA_R=64                   # richer adapter (vLLM: --max-lora-rank 64)
export FT_LORA_ALPHA=128              # keep alpha = 2*r
export FT_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# --- checkpointing (storage was expanded; LoRA checkpoints are megabytes) ----
export FT_SAVE_STRATEGY=epoch         # enables sft_lora.py --resume after a crash
export FT_SAVE_TOTAL_LIMIT=2

# --- dataset knobs ------------------------------------------------------------
export PREP_MAX_CHARS=12000           # match the longer context window
# PREP_NEGATIVES_FRAC defaults to 0.10; PREP_REPLAY_FRAC to 0.15

# --- cache --------------------------------------------------------------------
# /workspace sits on the pod's 879 GB local NVMe (612 GB free) - fast, plenty.
# It does NOT survive a pod rebuild, but a re-download costs minutes; precious
# artifacts (adapter, results.md, run_metadata.json) get copied to the
# pod-independent NFS share instead:  cp -r finetune/outputs /workspace/shared/
export HF_HOME=/workspace/hf-cache
mkdir -p "$HF_HOME"

echo "MI300X profile loaded: FT_BASE_MODEL=$FT_BASE_MODEL, r=$FT_LORA_R, seq=$FT_MAX_SEQ_LEN"
