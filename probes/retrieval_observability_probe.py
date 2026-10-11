#!/usr/bin/env python3
# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Live canary probe verifying Prometheus metrics and observer per-lane observability."""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def load_configs():
    home = Path.home()
    ov_conf_path = home / ".openviking" / "ov.conf"
    cli_conf_path = home / ".openviking" / "ovcli-antigravity.conf"

    if not ov_conf_path.exists():
        raise FileNotFoundError(f"Missing ov.conf at {ov_conf_path}")

    with open(ov_conf_path, "r", encoding="utf-8") as f:
        ov_conf = json.load(f)

    server_host = ov_conf.get("server", {}).get("host", "127.0.0.1")
    server_port = ov_conf.get("server", {}).get("port", 1933)
    root_api_key = ov_conf.get("server", {}).get("root_api_key", "")
    base_url = f"http://{server_host}:{server_port}"

    user_api_key = root_api_key
    if cli_conf_path.exists():
        try:
            with open(cli_conf_path, "r", encoding="utf-8") as f:
                cli_conf = json.load(f)
                user_api_key = cli_conf.get("api_key", root_api_key)
        except Exception:
            pass

    return {
        "base_url": base_url,
        "root_api_key": root_api_key,
        "user_api_key": user_api_key,
    }


def http_get(url: str, headers: dict = None) -> tuple[int, str]:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.status, resp.read().decode("utf-8")


def http_get_json(url: str, headers: dict = None) -> tuple[int, dict]:
    status, body = http_get(url, headers=headers)
    return status, json.loads(body)


def http_post(url: str, body: dict, headers: dict = None) -> tuple[int, dict]:
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=req_headers,
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def main():
    print("=" * 60)
    print("OpenViking Observability Hardening: Live Canary Probe")
    print("=" * 60)

    cfg = load_configs()
    base_url = cfg["base_url"]
    root_api_key = cfg["root_api_key"]
    user_api_key = cfg["user_api_key"]

    # 1. Health check
    print(f"\n[1/5] Checking server health at {base_url}/health...")
    status, health_data = http_get_json(f"{base_url}/health")
    if status != 200:
        print(f"FAILED: Health check returned HTTP {status}: {health_data}")
        return 1
    if not health_data.get("healthy"):
        print(f"FAILED: Server reported unhealthy: {health_data}")
        return 1
    print(f"  OK: Server healthy (version {health_data.get('version')})")

    # 2. Check initial observer status
    print("\n[2/5] Reading initial observer status...")
    status, obs_resp = http_get_json(
        f"{base_url}/api/v1/observer/retrieval?format=json",
        headers={"X-API-Key": root_api_key},
    )
    if status != 200 or obs_resp.get("status") != "ok":
        print(f"FAILED: Observer retrieval check failed: {obs_resp}")
        return 1
    initial_stats = obs_resp.get("result", {}).get("status", {})
    print(f"  Initial total queries: {initial_stats.get('total_queries', 0)}")
    print(f"  Initial rerank by lane: {initial_stats.get('rerank_by_lane', {})}")
    print(f"  Initial avg rerank latency by lane: {initial_stats.get('avg_rerank_latency_by_lane', {})}")

    # 3. Issue test queries on light and heavy lanes
    print("\n[3/5] Issuing test retrieval queries with lane routing...")
    # Light lane via search
    search_payload = {
        "query": "observability telemetry verification",
        "target_uri": "viking://user/monica/memories",
        "rerank_lane": "light",
        "limit": 3,
    }
    status, resp = http_post(
        f"{base_url}/api/v1/search/search",
        search_payload,
        headers={"X-API-Key": user_api_key},
    )
    if status != 200:
        print(f"FAILED: Search query failed with HTTP {status}: {resp}")
        return 1
    print("  OK: Light lane search completed successfully")

    # Heavy lane via find
    find_payload = {
        "query": "systematic optimization hardening canary",
        "target_uri": "viking://user/monica/memories",
        "rerank_lane": "heavy",
        "limit": 3,
    }
    status, resp = http_post(
        f"{base_url}/api/v1/search/find",
        find_payload,
        headers={"X-API-Key": user_api_key},
    )
    if status != 200:
        print(f"FAILED: Find query failed with HTTP {status}: {resp}")
        return 1
    print("  OK: Heavy lane find completed successfully")

    # 4. Verify Prometheus metrics contain lane dimension
    print("\n[4/5] Verifying Prometheus metrics for lane dimension...")
    status, metrics_text = http_get(f"{base_url}/metrics")
    if status != 200:
        print(f"FAILED: Failed to fetch /metrics (HTTP {status})")
        return 1

    rerank_metric_lines = [
        line for line in metrics_text.splitlines()
        if line.startswith("openviking_rerank_calls_total")
    ]
    print("  Prometheus rerank metric samples:")
    for line in rerank_metric_lines:
        print(f"    {line}")

    has_lane_label = any('lane=' in line for line in rerank_metric_lines)
    if not has_lane_label:
        print("FAILED: No 'lane' label found in openviking_rerank_calls_total metrics!")
        return 1
    print("  OK: Prometheus metrics successfully emit 'lane' dimension")

    # 5. Verify observer retrieval status contains avg_rerank_latency_by_lane
    print("\n[5/5] Verifying Observer retrieval per-lane latency tracking...")
    status, obs_resp = http_get_json(
        f"{base_url}/api/v1/observer/retrieval?format=json",
        headers={"X-API-Key": root_api_key},
    )
    if status != 200 or obs_resp.get("status") != "ok":
        print(f"FAILED: Observer retrieval check failed: {obs_resp}")
        return 1

    post_stats = obs_resp.get("result", {}).get("status", {})
    print(f"  Post-probe rerank by lane: {post_stats.get('rerank_by_lane', {})}")
    avg_latency = post_stats.get("avg_rerank_latency_by_lane")
    print(f"  Post-probe avg rerank latency by lane: {avg_latency}")

    if avg_latency is None:
        print("FAILED: 'avg_rerank_latency_by_lane' field missing from observer retrieval status!")
        return 1

    if not isinstance(avg_latency, dict):
        print(f"FAILED: Expected dict for 'avg_rerank_latency_by_lane', got {type(avg_latency)}")
        return 1

    print("  OK: Observer retrieval stats successfully tracks per-lane latency")

    print("\n" + "=" * 60)
    print("ALL CANARY CHECKS PASSED: Observability Hardening Verified")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
