#!/usr/bin/env python3
# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Golden Set Retrieval Quality Evaluation Probe."""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
import yaml


def load_config():
    home = Path.home()
    ov_conf_path = home / ".openviking" / "ov.conf"
    cli_conf_path = home / ".openviking" / "ovcli-monica.conf"
    if not cli_conf_path.exists():
        cli_conf_path = home / ".openviking" / "ovcli.conf"

    with open(ov_conf_path, "r", encoding="utf-8") as f:
        ov_conf = json.load(f)

    with open(cli_conf_path, "r", encoding="utf-8") as f:
        cli_conf = json.load(f)

    host = ov_conf.get("server", {}).get("host", "127.0.0.1")
    port = ov_conf.get("server", {}).get("port", 1933)
    api_key = cli_conf.get("api_key", "")

    return {
        "base_url": f"http://{host}:{port}",
        "api_key": api_key,
    }


def query_find(base_url: str, api_key: str, query: str, limit: int = 5):
    url = f"{base_url}/api/v1/search/find"
    payload = {
        "query": query,
        "target_uri": "viking://",
        "limit": limit,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
        res_list = data.get("result", {}).get("resources", [])
        mem_list = data.get("result", {}).get("memories", [])
        combined = sorted(res_list + mem_list, key=lambda x: x.get("score", 0.0), reverse=True)
        return combined[:limit]


def matches_expected(hit_uri: str, expected_uris: list[str]) -> bool:
    for exp in expected_uris:
        if hit_uri == exp:
            return True
        if exp.endswith("/") and hit_uri.startswith(exp):
            return True
        if exp.endswith("/.abstract.md") and hit_uri == exp.replace("/.abstract.md", ""):
            return True
    return False


def run_eval(golden_file: Path, output_file: Path = None):
    with open(golden_file, "r", encoding="utf-8") as f:
        golden_data = yaml.safe_load(f)

    cfg = load_config()
    base_url = cfg["base_url"]
    api_key = cfg["api_key"]

    queries = golden_data.get("queries", [])
    positive_results = []
    noise_results = []
    leakage_results = []

    print(f"Loaded {len(queries)} queries from {golden_file}")
    print(f"Evaluating against OpenViking at {base_url}...")
    print("-" * 80)
    print(f"{'ID':<4} | {'Cat':<9} | {'Top-1 Score':<11} | {'Hit?':<5} | {'Rank':<4} | {'Query'}")
    print("-" * 80)

    for item in queries:
        qid = item["id"]
        category = item.get("category", "")
        query_text = item["query"]
        expected_uris = item.get("expected_uris", [])

        t0 = time.time()
        try:
            hits = query_find(base_url, api_key, query_text, limit=5)
        except Exception as e:
            print(f"Error querying {qid}: {e}")
            hits = []

        top_score = hits[0].get("score", 0.0) if hits else 0.0
        hit_rank = None
        is_hit = False

        if expected_uris:
            for rank, hit in enumerate(hits, start=1):
                h_uri = hit.get("uri", "")
                if matches_expected(h_uri, expected_uris):
                    is_hit = True
                    hit_rank = rank
                    break

            positive_results.append({
                "id": qid,
                "category": category,
                "query": query_text,
                "top_score": top_score,
                "is_hit": is_hit,
                "rank": hit_rank,
                "hits_returned": len(hits),
            })
            rank_str = str(hit_rank) if hit_rank is not None else "-"
            print(f"{qid:<4} | {category:<9} | {top_score:<11.4f} | {str(is_hit):<5} | {rank_str:<4} | {query_text[:40]}")
        elif category == "leakage" or item.get("target_scope") == "private_memory":
            # Leakage evaluation: checks whether any private memory from other actors leaks.
            # Hits in shared viking://resources/ are public lore and excluded from leakage score.
            private_hits = [h for h in hits if not h.get("uri", "").startswith("viking://resources/")]
            shared_hits = [h for h in hits if h.get("uri", "").startswith("viking://resources/")]

            private_top_score = private_hits[0].get("score", 0.0) if private_hits else 0.0
            shared_top_score = shared_hits[0].get("score", 0.0) if shared_hits else 0.0

            leakage_results.append({
                "id": qid,
                "category": category,
                "query": query_text,
                "private_top_score": private_top_score,
                "shared_top_score": shared_top_score,
                "hits_returned": len(hits),
            })
            shared_info = f" [SHARED: {shared_top_score:.4f}]" if shared_top_score > 0 else " [ISOLATED]"
            print(f"{qid:<4} | {category:<9} | {private_top_score:<11.4f} | {'N/A':<5} | {'-':<4} | {query_text[:40]}{shared_info}")
        else:
            # Negative / Noise query
            noise_results.append({
                "id": qid,
                "category": category,
                "query": query_text,
                "top_score": top_score,
                "hits_returned": len(hits),
            })
            print(f"{qid:<4} | {category:<9} | {top_score:<11.4f} | {'N/A':<5} | {'-':<4} | {query_text[:40]} [NOISE]")

    # Metrics computation
    total_pos = len(positive_results)
    r1 = sum(1 for r in positive_results if r["rank"] == 1) / total_pos if total_pos else 0
    r3 = sum(1 for r in positive_results if r["rank"] is not None and r["rank"] <= 3) / total_pos if total_pos else 0
    r5 = sum(1 for r in positive_results if r["rank"] is not None and r["rank"] <= 5) / total_pos if total_pos else 0
    mrr = sum((1.0 / r["rank"]) for r in positive_results if r["rank"] is not None) / total_pos if total_pos else 0

    pos_avg_score = sum(r["top_score"] for r in positive_results) / total_pos if total_pos else 0
    noise_avg_score = sum(r["top_score"] for r in noise_results) / len(noise_results) if noise_results else 0
    leakage_avg_score = sum(r["private_top_score"] for r in leakage_results) / len(leakage_results) if leakage_results else 0
    score_sep = pos_avg_score - noise_avg_score

    metrics = {
        "total_queries": len(queries),
        "positive_queries": total_pos,
        "noise_queries": len(noise_results),
        "leakage_queries": len(leakage_results),
        "recall_at_1": round(r1, 4),
        "recall_at_3": round(r3, 4),
        "recall_at_5": round(r5, 4),
        "mrr": round(mrr, 4),
        "positive_avg_score": round(pos_avg_score, 4),
        "noise_avg_score": round(noise_avg_score, 4),
        "leakage_avg_score": round(leakage_avg_score, 4),
        "score_separation": round(score_sep, 4),
    }

    print("=" * 80)
    print("EVALUATION METRICS SUMMARY")
    print(f"  Recall@1:           {metrics['recall_at_1'] * 100:.1f}% ({sum(1 for r in positive_results if r['rank'] == 1)}/{total_pos})")
    print(f"  Recall@3:           {metrics['recall_at_3'] * 100:.1f}% ({sum(1 for r in positive_results if r['rank'] is not None and r['rank'] <= 3)}/{total_pos})")
    print(f"  Recall@5:           {metrics['recall_at_5'] * 100:.1f}% ({sum(1 for r in positive_results if r['rank'] is not None and r['rank'] <= 5)}/{total_pos})")
    print(f"  MRR:                {metrics['mrr']:.4f}")
    print(f"  Pos Top-1 Avg:      {metrics['positive_avg_score']:.4f}")
    print(f"  Noise Top-1 Avg:    {metrics['noise_avg_score']:.4f}")
    if leakage_results:
        print(f"  Leakage Top-1 Avg:  {metrics['leakage_avg_score']:.4f}")
    print(f"  Score Separation:   {metrics['score_separation']:.4f}")
    print("=" * 80)

    if output_file:
        output_payload = {
            "metrics": metrics,
            "positive_results": positive_results,
            "noise_results": noise_results,
            "leakage_results": leakage_results,
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)
        print(f"Detailed results saved to {output_file}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=str(Path(__file__).parent / "golden_set.yaml"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run_eval(Path(args.golden), Path(args.output) if args.output else None)
