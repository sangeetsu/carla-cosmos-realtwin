#!/usr/bin/env python3
"""Check Hugging Face access and download every model the jobs need (exact revisions used by the tested Cosmos commit).
Run by setup.sh inside the Cosmos venv. Safe to re-run: cached files are skipped."""
import os, sys, time
from huggingface_hub import hf_hub_download, snapshot_download

GATED = ["nvidia/Cosmos-Transfer2.5-2B", "nvidia/Cosmos-Guardrail1", "nvidia/Cosmos-Reason1-7B", "nvidia/Cosmos-Predict2.5-2B"]
def log(m): print(f"[predownload {time.strftime('%H:%M:%S')}] {m}", flush=True)

bad = []
for r in GATED:
    try: hf_hub_download(r, ".gitattributes")
    except Exception as e: bad.append(f"{r} ({type(e).__name__})")
if bad:
    log("ERROR: no access to: " + ", ".join(bad))
    log("Fix: log in to huggingface.co, open each page above, accept the NVIDIA license, then either `export HF_TOKEN=<your read token>`")
    log("     or run `" + os.environ.get("WORK", "work") + "/cosmos-transfer2.5/.venv/bin/hf auth login`, and re-run setup.sh.")
    sys.exit(1)
log("Hugging Face access OK for all gated NVIDIA repos")

# Control-net checkpoints, through Cosmos's own registry (same files and revisions it fetches at run time)
sys.path.insert(0, os.environ["COSMOS"])
import cosmos_transfer2.config as C
for k, v in C.MODEL_CHECKPOINTS.items():
    name = k.name if hasattr(k, "name") else str(k)
    if k.variant.value in ("edge", "depth", "seg", "vis") and (not getattr(k, "distilled", False) or k.variant.value == "edge"):
        log(f"checkpoint {name}"); v.download()

# Text encoder, tokenizer and guardrail models (revisions as logged by the tested runs)
log("text encoder nvidia/Cosmos-Reason1-7B"); snapshot_download("nvidia/Cosmos-Reason1-7B", revision="3210bec0495fdc7a8d3dbb8d58da5711eab4b423")
log("tokenizer nvidia/Cosmos-Predict2.5-2B"); hf_hub_download("nvidia/Cosmos-Predict2.5-2B", "tokenizer.pth", revision="f176dc95b4a70f53ce01c4b302851595e7322b00")
log("guardrail nvidia/Cosmos-Guardrail1"); snapshot_download("nvidia/Cosmos-Guardrail1", revision="d6d4bfa899a71454a700907664f3e88f503950cf")
for r in ["Qwen/Qwen3Guard-Gen-0.6B", "google/siglip-so400m-patch14-384"]:
    log(f"guardrail helper {r}"); snapshot_download(r)
# On-the-fly depth / segmentation for the multi-control job (J2); failure here only disables J2
for r in ["depth-anything/Video-Depth-Anything-Large", "IDEA-Research/grounding-dino-base", "facebook/sam2-hiera-large"]:
    try: log(f"multi-control helper {r}"); snapshot_download(r)
    except Exception as e: log(f"WARNING: could not download {r} ({type(e).__name__}); job J2 may fail, the others are unaffected")
log("all downloads done; cache " + os.environ.get("HF_HUB_CACHE", "?"))
