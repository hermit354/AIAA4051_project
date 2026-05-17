"""Check a Hugging Face token without printing the token itself.

The script is intended for local troubleshooting when `huggingface_hub.login`
returns 401. It asks for the token twice, reports non-secret format checks, and
calls `whoami` with the token. It can optionally save the token to the local
Hugging Face cache after a successful check.
"""

from __future__ import annotations

import argparse
import getpass

import requests
from huggingface_hub import HfFolder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true", help="Save the token locally after validation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = getpass.getpass("Paste Hugging Face token: ").strip()
    token_again = getpass.getpass("Paste it again: ").strip()

    print(f"Length: {len(token)}")
    print(f"Starts with hf_: {token.startswith('hf_')}")
    print(f"Two entries match: {token == token_again}")
    print(f"Contains whitespace: {any(ch.isspace() for ch in token)}")

    if token != token_again:
        raise SystemExit("The two pasted tokens do not match. Copy it again from Hugging Face.")
    if not token.startswith("hf_"):
        raise SystemExit("The token does not start with hf_. Make sure you copied the token value.")
    if any(ch.isspace() for ch in token):
        raise SystemExit("The token contains whitespace. Copy with the Hugging Face copy button.")

    headers = {"Authorization": f"Bearer {token}"}
    authenticated_user = None
    for endpoint in ("https://huggingface.co/api/whoami-v2", "https://huggingface.co/api/whoami"):
        response = requests.get(endpoint, headers=headers, timeout=20)
        print(f"{endpoint}: HTTP {response.status_code}")
        if response.status_code == 200 and endpoint.endswith("whoami-v2"):
            payload = response.json()
            authenticated_user = payload.get("name") or payload.get("fullname") or payload.get("email")
        elif response.status_code != 200:
            print(f"  Response: {response.text[:200]}")

    if not authenticated_user:
        raise SystemExit("The token did not authenticate against whoami-v2.")

    print(f"Authenticated as: {authenticated_user}")

    if args.save:
        HfFolder.save_token(token)
        print("Token saved to the local Hugging Face cache.")


if __name__ == "__main__":
    main()
