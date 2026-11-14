"""Deployment smoke test for the public RAG API.

Usage:
  python scripts/smoke_test.py https://your-api.onrender.com
  python scripts/smoke_test.py https://your-api.onrender.com --token YOUR_API_AUTH_TOKEN

The token is read from the command line only for this local invocation; never commit it.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request


def request(url: str, token: str | None = None):
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {"detail": body[:500]}
        return exc.code, payload


def main():
    parser = argparse.ArgumentParser(description="Smoke-test a deployed RAG Failure Diagnostics API.")
    parser.add_argument("base_url", help="Backend base URL, e.g. https://your-api.onrender.com")
    parser.add_argument("--token", default="", help="Optional Render-generated API_AUTH_TOKEN")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    failures = 0

    checks = [
        ("root", f"{base}/", 200, None),
        ("health", f"{base}/health", 200, None),
        ("protected documents endpoint", f"{base}/api/documents", 200 if args.token else 401,
         args.token or None),
        ("failure taxonomy", f"{base}/api/diagnostics/patterns", 200 if args.token else 401,
         args.token or None),
    ]
    for name, url, expected, token in checks:
        try:
            status, payload = request(url, token)
            passed = status == expected
            print(f"[{'PASS' if passed else 'FAIL'}] {name}: HTTP {status} (expected {expected})")
            if not passed:
                print(json.dumps(payload, indent=2)[:1000])
                failures += 1
        except Exception as exc:
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            failures += 1

    if failures:
        print(f"\\n{failures} smoke check(s) failed.")
        return 1
    print("\\nAll deployment smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
