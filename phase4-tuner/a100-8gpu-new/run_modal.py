"""
Modal app: Phase 4 RL-based tuner on 8x A100.
Runs the Phase 3 iteration proxy under the RL bandit tuner plugin and
logs iteration times and rewards. Job name: browser-networking-test.
"""

import os
import subprocess
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Image: CUDA 12 + PyTorch, plus NCCL headers and RL tuner plugin build.
rl_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("wget", "build-essential")
    .run_commands("pip install --upgrade pip")
    .pip_install("torch")
    .add_local_dir(REPO_ROOT, remote_path="/repo", copy=True)
    .run_commands(
        # Build RL bandit tuner plugin shared library using local tuner headers.
        "cd /repo/phase4-tuner && "
        "gcc -fPIC -shared -I. "
        "-o /usr/local/lib/libnccl-tuner-rl-bandit.so "
        "rl_bandit_tuner_plugin.c",
    )
)

volume = modal.Volume.from_name("cs244c-nccl-results", create_if_missing=True)
VOLUME_PATH = "/results"

app = modal.App("browser-networking-tests")

PROXY_SCRIPT = "/repo/phase3-iteration-proxy/a100-8gpu-new/iteration_proxy.py"


@app.function(
    name="browser-networking-test",
    image=rl_image,
    gpu="A100:8",
    timeout=1800,
    volumes={VOLUME_PATH: volume},
)
def run_iteration_proxy_with_rl_tuner(iters: int = 100, warmup: int = 10):
    """Run iteration proxy once under the RL bandit tuner and save outputs."""
    results_dir = Path(VOLUME_PATH)
    results_dir.mkdir(parents=True, exist_ok=True)

    reward_file = results_dir / "rl_bandit_rewards.log"
    out_file = results_dir / "iteration_times_rl_bandit.txt"

    env = {
        **os.environ,
        "NCCL_TUNER_PLUGIN": "libnccl-tuner-rl-bandit.so",
        "NCCL_TUNER_REWARD_FILE": str(reward_file),
        # Optional: tune exploration rate
        "NCCL_TUNER_EPS": os.environ.get("NCCL_TUNER_EPS", "0.1"),
        # Helpful debug from NCCL
        "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "INFO"),
    }

    cmd = [
        "python",
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=8",
        "--standalone",
        PROXY_SCRIPT,
        "--iters",
        str(iters),
        "--warmup",
        str(warmup),
        "--out",
        str(out_file),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd="/repo")
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"iteration_proxy exited {result.returncode}")

    print(result.stdout, flush=True)
    if out_file.is_file():
        lines = out_file.read_text().strip().split("\n")
    else:
        lines = []

    volume.commit()
    return {"iteration_times": lines, "reward_file": str(reward_file)}


@app.local_entrypoint()
def main(iters: int = 100, warmup: int = 10):
    """Run the RL bandit tuner experiment and copy results locally."""
    out = run_iteration_proxy_with_rl_tuner.remote(iters=iters, warmup=warmup)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    times_path = results_dir / "iteration_times_rl_bandit.txt"
    times_path.write_text("\n".join(out["iteration_times"]) + "\n")

    # Reward log will live on the remote volume; copy path info for reference.
    meta_path = results_dir / "rl_bandit_metadata.txt"
    meta_path.write_text(f"Remote reward file: {out['reward_file']}\n")

    print(f"Wrote {times_path}")
    print(f"Wrote {meta_path}")
    print("Done.")

