"""
Deterministic Benchmark Dataset Generator and Scorer for Long-Context Recall.
Implements Multi-Depth Needle-in-a-Haystack (MD-NIAH) with single-fact and multi-fact relational questions.
"""

import re
import json
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from transformers import PreTrainedTokenizer


DISTRACTOR_TEMPLATES = [
    (
        "Cluster telemetry report: Node {node_id} reported an average CPU utilization of {cpu_pct}% "
        "over the previous 15-minute measurement window. Memory pressure index remained nominal at {mem_val} GiB. "
        "Ingress packet queue length was recorded as {queue_len} packets with zero drop anomalies. "
        "Storage volume {vol_id} maintained an I/O operational throughput of {io_ops} IOPS with a read latency of {lat_ms} ms. "
        "Network interface eth0 observed {rx_mb} MB received and {tx_mb} MB transmitted without CRC checksum mismatches."
    ),
    (
        "Service mesh traffic advisory: Envoy proxy instances in zone {zone_id} routed {req_count} requests per second "
        "with an HTTP 200 success ratio of {success_pct}%. Upstream service latency p95 settled at {p95} ms, "
        "while p99 latency was contained at {p99} ms. Circuit breaker trip count was 0 across all worker pods. "
        "Active TLS sessions negotiated using cipher suite TLS_AES_256_GCM_SHA384 with session resumption rate {resume_pct}%."
    ),
    (
        "Database maintenance dispatch: Relational shard {shard_id} completed table vacuuming and b-tree index optimization. "
        "Dead tuple cleanup reclaimed {dead_mb} MB across {scanned_pages} disk pages. Wal sequence buffer flush rate "
        "averaged {flush_kb} KB/sec. Connection pool pool-{pool_id} serviced {conn_count} concurrent client handles "
        "with an average acquisition wait time of {acq_ms} microseconds. Cache hit ratio registered at {hit_ratio}%."
    ),
    (
        "Container orchestration ledger: Deployment {deploy_id} scaled replica count from {old_rep} to {new_rep} pods "
        "in response to horizontal pod autoscaler signal {hpa_signal}. Container image digest sha256:{digest} passed "
        "cryptographic signature verification. Readiness probes succeeded on port {probe_port} after an initial delay of {init_delay} s. "
        "CoreDNS resolution latency across pod subnet remained below {dns_lat} ms."
    ),
    (
        "Security compliance audit: Key management service performed scheduled key rotation for ring-{ring_id}. "
        "Access control tokens validated against OAuth identity provider with token cache TTL set to {ttl_s} seconds. "
        "Firewall rule-table {rule_id} inspected {inspected_packets} ingress frames, dropping {drop_count} unmatched fragments. "
        "Audit log hash chaining verified integrity with zero sequence gaps across event partition {part_id}."
    ),
]


@dataclass
class NeedleFact:
    fact_id: str
    fact_type: str  # e.g., "project_code", "database_port", "deployment_region", "failover_threshold"
    key_name: str
    key_value: str
    sentence: str


@dataclass
class BenchmarkItem:
    item_id: str
    target_context_length: int
    actual_token_length: int
    depth_category: str  # "start", "early_middle", "middle", "late_middle", "end"
    depth_fraction: float  # e.g. 0.05, 0.25, 0.50, 0.75, 0.95
    query_type: str  # "single_fact" or "multi_fact"
    needles: List[Dict[str, Any]]
    document: str
    question: str
    ground_truth_answers: List[str]
    seed: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_distractor_paragraph(rng: random.Random) -> str:
    """Generates a pseudo-random technical distractor paragraph using template substitution."""
    tmpl = rng.choice(DISTRACTOR_TEMPLATES)
    values = {
        "node_id": f"node-{rng.randint(100, 999)}",
        "cpu_pct": round(rng.uniform(10.0, 85.0), 1),
        "mem_val": round(rng.uniform(4.0, 32.0), 2),
        "queue_len": rng.randint(0, 50),
        "vol_id": f"vol-{rng.randint(1000, 9999)}",
        "io_ops": rng.randint(200, 4500),
        "lat_ms": round(rng.uniform(0.5, 8.0), 2),
        "rx_mb": rng.randint(50, 1500),
        "tx_mb": rng.randint(50, 1500),
        "zone_id": f"zone-{rng.choice(['a', 'b', 'c', 'd'])}-{rng.randint(1, 9)}",
        "req_count": rng.randint(500, 12000),
        "success_pct": round(rng.uniform(99.1, 99.99), 2),
        "p95": round(rng.uniform(5.0, 45.0), 1),
        "p99": round(rng.uniform(15.0, 95.0), 1),
        "resume_pct": round(rng.uniform(70.0, 95.0), 1),
        "shard_id": f"shard-{rng.randint(1, 16)}",
        "dead_mb": rng.randint(10, 500),
        "scanned_pages": rng.randint(1000, 50000),
        "flush_kb": rng.randint(100, 5000),
        "pool_id": rng.randint(1, 8),
        "conn_count": rng.randint(10, 200),
        "acq_ms": rng.randint(10, 800),
        "hit_ratio": round(rng.uniform(94.0, 99.8), 1),
        "deploy_id": f"deploy-svc-{rng.randint(10, 99)}",
        "old_rep": rng.randint(2, 8),
        "new_rep": rng.randint(4, 16),
        "hpa_signal": f"sig-{rng.randint(100, 999)}",
        "digest": f"{rng.randint(10000000, 99999999):x}{rng.randint(10000000, 99999999):x}",
        "probe_port": rng.choice([8080, 8443, 9090, 5000]),
        "init_delay": rng.randint(5, 30),
        "dns_lat": round(rng.uniform(0.2, 2.5), 2),
        "ring_id": f"ring-{rng.randint(1, 5)}",
        "ttl_s": rng.choice([3600, 7200, 14400]),
        "rule_id": f"sec-rule-{rng.randint(100, 999)}",
        "inspected_packets": rng.randint(10000, 900000),
        "drop_count": rng.randint(0, 15),
        "part_id": f"part-{rng.randint(0, 7)}",
    }
    return tmpl.format(**values)


def normalize_text(text: str) -> str:
    """Normalize text for deterministic string matching."""
    text = text.lower()
    text = re.sub(r"[^\w\s\-\.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def score_response(prediction: str, ground_truths: List[str]) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Evaluates response against ground truth answers.
    Returns:
        (is_fully_correct, score_fraction, details)
    """
    norm_pred = normalize_text(prediction)
    matches = []
    for gt in ground_truths:
        norm_gt = normalize_text(gt)
        # Check direct substring inclusion
        matched = norm_gt in norm_pred
        matches.append(matched)

    all_matched = all(matches) if matches else False
    score_fraction = (sum(matches) / len(matches)) if matches else 0.0

    return all_matched, score_fraction, {
        "ground_truths": ground_truths,
        "matches": matches,
        "score_fraction": score_fraction,
        "is_fully_correct": all_matched,
    }


class BenchmarkGenerator:
    """
    Generates reproducible long-context evaluation datasets with facts embedded at precise document depths.
    """

    DEPTH_SPEC = [
        ("start", 0.05),
        ("early_middle", 0.25),
        ("middle", 0.50),
        ("late_middle", 0.75),
        ("end", 0.95),
    ]

    def __init__(self, tokenizer: PreTrainedTokenizer, base_seed: int = 42):
        self.tokenizer = tokenizer
        self.base_seed = base_seed

    def generate_needle_set(self, rng: random.Random, item_prefix: str) -> Tuple[NeedleFact, NeedleFact]:
        """Generates two correlated high-entropy facts for single and multi-fact questions."""
        proj_num = rng.randint(1000, 9999)
        proj_code = f"PROJ-{proj_num}"
        db_port = str(rng.randint(5000, 9999))
        region = f"{rng.choice(['us-east', 'eu-west', 'ap-south', 'sa-east'])}-{rng.randint(1, 3)}"
        threshold = f"{rng.randint(70, 99)}.{rng.randint(1, 9)}%"

        fact1 = NeedleFact(
            fact_id=f"{item_prefix}_fact1",
            fact_type="project_and_port",
            key_name="project_code_and_port",
            key_value=f"{proj_code}:{db_port}",
            sentence=(
                f"Critical Infrastructure Record: System project code is {proj_code}. "
                f"The designated primary database operates on port {db_port}."
            ),
        )
        fact2 = NeedleFact(
            fact_id=f"{item_prefix}_fact2",
            fact_type="region_and_threshold",
            key_name="region_and_threshold",
            key_value=f"{region}:{threshold}",
            sentence=(
                f"Failover Architecture Directive: Project {proj_code} deployment region is strictly assigned to {region}. "
                f"The cluster automated failover threshold is configured at exactly {threshold}."
            ),
        )
        return fact1, fact2

    def build_item(
        self,
        item_id: str,
        target_context_length: int,
        depth_category: str,
        depth_fraction: float,
        query_type: str,
        seed: int,
    ) -> BenchmarkItem:
        """Constructs a single long-context document item with precise token length and needle insertion."""
        rng = random.Random(seed)
        fact1, fact2 = self.generate_needle_set(rng, item_id)

        if query_type == "single_fact":
            target_needles = [fact1]
            question = (
                f"What is the designated primary database port for project {fact1.key_value.split(':')[0]}? "
                "Answer concisely with only the exact port number:"
            )
            ground_truth = [fact1.key_value.split(":")[1]]
        else:  # multi_fact relational query
            target_needles = [fact1, fact2]
            question = (
                f"For project {fact1.key_value.split(':')[0]}, state its assigned deployment region "
                "and its primary database port. Answer concisely:"
            )
            ground_truth = [
                fact2.key_value.split(":")[0],  # region
                fact1.key_value.split(":")[1],  # port
            ]

        preamble = (
            "Enterprise Architecture Documentation and Systems Verification Specification.\n"
            "This document contains operational configurations, cluster telemetry, and deployment rules.\n"
            "Read all sections carefully to answer the inquiry at the end.\n\n"
        )
        query_suffix = f"\n\n[Inquiry]\n{question}\nAnswer:"

        preamble_tokens = len(self.tokenizer.encode(preamble, add_special_tokens=False))
        suffix_tokens = len(self.tokenizer.encode(query_suffix, add_special_tokens=False))
        needle_text = f"\n\n{fact1.sentence} {fact2.sentence}\n\n"
        needle_tokens = len(self.tokenizer.encode(needle_text, add_special_tokens=False))

        # Exact budget for distractors
        target_distractor_tokens = max(50, target_context_length - preamble_tokens - suffix_tokens - needle_tokens)

        # Generate paragraphs up to target_distractor_tokens
        distractor_paras: List[str] = []
        accumulated_tokens = 0
        while accumulated_tokens < target_distractor_tokens:
            para = generate_distractor_paragraph(rng)
            p_ids = self.tokenizer.encode(para, add_special_tokens=False)
            if accumulated_tokens + len(p_ids) > target_distractor_tokens:
                # Trim last paragraph to fit exact remaining tokens
                needed = target_distractor_tokens - accumulated_tokens
                trimmed_para = self.tokenizer.decode(p_ids[:needed], skip_special_tokens=True)
                distractor_paras.append(trimmed_para)
                accumulated_tokens += needed
                break
            else:
                distractor_paras.append(para)
                accumulated_tokens += len(p_ids)

        # Insert needle at specified depth fraction within distractor paragraphs
        insert_idx = int(round(len(distractor_paras) * depth_fraction))
        insert_idx = max(0, min(len(distractor_paras), insert_idx))

        before_needle = "\n\n".join(distractor_paras[:insert_idx])
        after_needle = "\n\n".join(distractor_paras[insert_idx:])

        parts = [preamble.strip()]
        if before_needle.strip():
            parts.append(before_needle.strip())
        parts.append(needle_text.strip())
        if after_needle.strip():
            parts.append(after_needle.strip())

        body_text = "\n\n".join(parts)
        full_text = body_text + query_suffix
        actual_tokens = len(self.tokenizer.encode(full_text, add_special_tokens=True))

        return BenchmarkItem(
            item_id=item_id,
            target_context_length=target_context_length,
            actual_token_length=actual_tokens,
            depth_category=depth_category,
            depth_fraction=depth_fraction,
            query_type=query_type,
            needles=[asdict(f) for f in target_needles],
            document=full_text,
            question=question,
            ground_truth_answers=ground_truth,
            seed=seed,
        )

    def generate_benchmark_suite(
        self,
        context_lengths: List[int] = [1024, 2048, 4096, 8192],
    ) -> List[BenchmarkItem]:
        """Generates comprehensive evaluation suite across context lengths, depths, and query types."""
        items: List[BenchmarkItem] = []
        counter = 1

        for ctx_len in context_lengths:
            for depth_cat, depth_frac in self.DEPTH_SPEC:
                for q_type in ["single_fact", "multi_fact"]:
                    item_id = f"item_L{ctx_len}_{depth_cat}_{q_type}_{counter:03d}"
                    seed = self.base_seed + counter * 101
                    item = self.build_item(
                        item_id=item_id,
                        target_context_length=ctx_len,
                        depth_category=depth_cat,
                        depth_fraction=depth_frac,
                        query_type=q_type,
                        seed=seed,
                    )
                    items.append(item)
                    counter += 1

        return items

    def generate_pilot_suite(self) -> List[BenchmarkItem]:
        """Generates 5-sample pilot suite covering all 5 depths at 1024 tokens for quick testing."""
        items: List[BenchmarkItem] = []
        for idx, (depth_cat, depth_frac) in enumerate(self.DEPTH_SPEC):
            item_id = f"pilot_L1024_{depth_cat}_{idx+1:02d}"
            seed = self.base_seed + (idx + 1) * 7
            item = self.build_item(
                item_id=item_id,
                target_context_length=1024,
                depth_category=depth_cat,
                depth_fraction=depth_frac,
                query_type="single_fact" if idx % 2 == 0 else "multi_fact",
                seed=seed,
            )
            items.append(item)
        return items
