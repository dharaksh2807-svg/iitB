"""
Memory and Latency Metrics Instrumentation.
Implements exact physical tensor size calculations and multi-tier memory tracking.
"""

import sys
import time
import resource
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
import torch
import psutil


def get_process_rss_bytes() -> int:
    """Return the current resident set size (RSS) of the process in bytes."""
    process = psutil.Process()
    return process.memory_info().rss


def get_peak_rss_bytes() -> int:
    """
    Return the peak resident set size of the process in bytes.
    Note: On macOS Darwin, getrusage().ru_maxrss is in bytes.
    On Linux, getrusage().ru_maxrss is in kilobytes.
    """
    ru_max = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(ru_max)
    else:
        return int(ru_max * 1024)


def get_device_allocated_bytes(device: torch.device) -> int:
    """Return device-allocated memory in bytes if supported by accelerator."""
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
        return torch.mps.current_allocated_memory()
    return 0


def get_device_driver_bytes(device: torch.device) -> int:
    """Return device driver-allocated memory in bytes if supported."""
    if device.type == "cuda":
        return torch.cuda.memory_reserved(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "driver_allocated_memory"):
        return torch.mps.driver_allocated_memory()
    return 0


def calculate_kv_cache_bytes(past_key_values: Any) -> Tuple[int, int, Dict[str, Any]]:
    """
    Directly calculate the physical memory occupied by retained Key and Value cache tensors.
    
    Formula:
        Total Bytes = sum_l (numel(K_l) * element_size(K_l) + numel(V_l) * element_size(V_l))
    
    Returns:
        (total_bytes, layers_measured, layer_details)
    """
    if past_key_values is None:
        return 0, 0, {}

    total_bytes = 0
    layers_measured = 0
    layer_details: List[Dict[str, Any]] = []

    # Modern Transformers DynamicCache (has .layers attribute)
    if hasattr(past_key_values, "layers"):
        for idx, layer in enumerate(past_key_values.layers):
            k = getattr(layer, "keys", None)
            v = getattr(layer, "values", None)
            if k is not None and v is not None:
                k_bytes = k.numel() * k.element_size()
                v_bytes = v.numel() * v.element_size()
                layer_bytes = k_bytes + v_bytes
                total_bytes += layer_bytes
                layers_measured += 1
                layer_details.append({
                    "layer": idx,
                    "k_shape": list(k.shape),
                    "v_shape": list(v.shape),
                    "k_dtype": str(k.dtype),
                    "v_dtype": str(v.dtype),
                    "bytes": layer_bytes
                })
    # Legacy tuple of tuples: ((k1, v1), (k2, v2), ...)
    elif isinstance(past_key_values, (tuple, list)):
        for idx, layer_kv in enumerate(past_key_values):
            if isinstance(layer_kv, (tuple, list)) and len(layer_kv) >= 2:
                k, v = layer_kv[0], layer_kv[1]
                if isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor):
                    k_bytes = k.numel() * k.element_size()
                    v_bytes = v.numel() * v.element_size()
                    layer_bytes = k_bytes + v_bytes
                    total_bytes += layer_bytes
                    layers_measured += 1
                    layer_details.append({
                        "layer": idx,
                        "k_shape": list(k.shape),
                        "v_shape": list(v.shape),
                        "k_dtype": str(k.dtype),
                        "v_dtype": str(v.dtype),
                        "bytes": layer_bytes
                    })

    meta = {
        "layers_measured": layers_measured,
        "first_layer": layer_details[0] if layer_details else None
    }
    return total_bytes, layers_measured, meta


def theoretical_kv_cache_bytes(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    element_size: int,
    seq_len: int,
    batch_size: int = 1
) -> int:
    """
    Theoretical KV-cache formula for Grouped-Query Attention (GQA):
        Bytes = batch_size * 2 * num_layers * num_kv_heads * seq_len * head_dim * element_size
    """
    return batch_size * 2 * num_layers * num_kv_heads * seq_len * head_dim * element_size


@dataclass
class MemoryCheckpoint:
    name: str
    timestamp: float
    process_rss_bytes: int
    peak_rss_bytes: int
    device_allocated_bytes: int
    device_driver_bytes: int
    kv_cache_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": round(self.timestamp, 4),
            "process_rss_mib": round(self.process_rss_bytes / (1024**2), 2),
            "peak_rss_mib": round(self.peak_rss_bytes / (1024**2), 2),
            "device_allocated_mib": round(self.device_allocated_bytes / (1024**2), 2),
            "device_driver_mib": round(self.device_driver_bytes / (1024**2), 2),
            "kv_cache_mib": round(self.kv_cache_bytes / (1024**2), 4),
        }


class MemoryTracker:
    """
    Tracks memory across inference checkpoints to clearly separate
    model-weight overhead, prefill peaks, cache retention, and decoding.
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.checkpoints: List[MemoryCheckpoint] = []

    def record(self, name: str, past_key_values: Any = None) -> MemoryCheckpoint:
        kv_bytes, _, _ = calculate_kv_cache_bytes(past_key_values)
        cp = MemoryCheckpoint(
            name=name,
            timestamp=time.time(),
            process_rss_bytes=get_process_rss_bytes(),
            peak_rss_bytes=get_peak_rss_bytes(),
            device_allocated_bytes=get_device_allocated_bytes(self.device),
            device_driver_bytes=get_device_driver_bytes(self.device),
            kv_cache_bytes=kv_bytes,
        )
        self.checkpoints.append(cp)
        return cp

    def summary(self) -> Dict[str, Any]:
        return {cp.name: cp.to_dict() for cp in self.checkpoints}
