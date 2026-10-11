"""Publishes the privately trained release models to the Hugging Face Hub (requires `hf auth login`).

usage: python publish_models.py <results dir> <namespace>
"""
import os
import sys

from huggingface_hub import HfApi


def main(res, namespace):
    api = HfApi()
    print("signed in as", api.whoami()["name"])
    for eps in (8, 3, 1):
        folder = os.path.join(res, f"pub_dp_eps{eps}_seed42", "release_model")
        repo = f"{namespace}/luganda-pii-dp-eps{eps}"
        api.create_repo(repo, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=folder, repo_id=repo, repo_type="model",
                          commit_message=f"Luganda PII detector trained with DP-SGD (epsilon={eps}, delta=1e-5)")
        print("published", f"https://huggingface.co/{repo}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
