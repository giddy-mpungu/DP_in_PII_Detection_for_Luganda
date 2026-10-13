"""Publishes the privately trained release models to the Hugging Face Hub (requires `hf auth login`).

usage: python publish_models.py <results dir> <namespace> [v1|v2]
  v1: Experiment C models (pub_dp_eps{8,3,1}_seed42)            -> <namespace>/luganda-pii-dp-eps{8,3,1}
  v2: Experiment E models (big_dp_eps{8,3,1,0.5}_seed42)        -> <namespace>/luganda-pii-dp-v2-eps{8,3,1,0.5}
"""
import os
import sys

from huggingface_hub import HfApi

RELEASES = {
    "v1": [(e, f"pub_dp_eps{e}_seed42", f"luganda-pii-dp-eps{e}") for e in ("8", "3", "1")],
    "v2": [(e, f"big_dp_eps{e}_seed42", f"luganda-pii-dp-v2-eps{e}") for e in ("8", "3", "1", "0.5")],
}


def main(res, namespace, version="v1"):
    api = HfApi()
    print("signed in as", api.whoami()["name"])
    for eps, run, name in RELEASES[version]:
        folder = os.path.join(res, run, "release_model")
        repo = f"{namespace}/{name}"
        api.create_repo(repo, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=folder, repo_id=repo, repo_type="model",
                          commit_message=f"Luganda PII detector trained with DP-SGD (epsilon={eps}, delta=1e-5)")
        print("published", f"https://huggingface.co/{repo}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
