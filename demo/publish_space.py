"""Publishes demo/site/ as a static Hugging Face Space (requires `hf auth login` as the account owner).

usage: python demo/publish_space.py <namespace/space-name> --yes
"""
import os
import sys

from huggingface_hub import HfApi

SITE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site")


def main(repo):
    api = HfApi()
    print("signed in as", api.whoami()["name"])
    api.create_repo(repo, repo_type="space", space_sdk="static", exist_ok=True)
    # the browser models live in the model repositories (onnx/), not in the Space
    api.upload_folder(folder_path=SITE, repo_id=repo, repo_type="space", ignore_patterns=["models/**"],
                      commit_message="Luganda PII detector demo (DP models, in-browser)")
    print("published", f"https://huggingface.co/spaces/{repo}")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[2] != "--yes":
        sys.exit(__doc__)
    main(sys.argv[1])
