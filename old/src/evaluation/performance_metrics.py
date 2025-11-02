"""
Performance Metrics for RAG Systems

This module provides utilities for measuring and profiling performance
of classical and quantum RAG systems.
"""

import time
import psutil
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from contextlib import contextmanager
from dataclasses import dataclass
import json


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    latency: float
    memory_usage: float
    cpu_usage: float
    throughput: float
    error_rate: float


class LatencyProfiler:
    """
    Profiler for measuring latency of RAG operations.
    """
    
    def __init__(self):
        self.measurements = {}
        
    @contextmanager
    def measure(self, operation_name: str):
        """Context manager for measuring operation latency."""
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            latency = end_time - start_time
            
            if operation_name not in self.measurements:
                self.measurements[operation_name] = []
            self.measurements[operation_name].append(latency)
    
    def get_stats(self, operation_name: str) -> Dict[str, float]:
        """Get statistics for an operation."""
        if operation_name not in self.measurements:
            return {}
        
        measurements = self.measurements[operation_name]
        return {
            "mean_latency": np.mean(measurements),
            "std_latency": np.std(measurements),
            "min_latency": np.min(measurements),
            "max_latency": np.max(measurements),
            "median_latency": np.median(measurements),
            "p95_latency": np.percentile(measurements, 95),
            "p99_latency": np.percentile(measurements, 99),
            "total_calls": len(measurements)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        return {op: self.get_stats(op) for op in self.measurements.keys()}
    
    def reset(self):
        """Reset all measurements."""
        self.measurements.clear()


class MemoryProfiler:
    """
    Profiler for measuring memory usage.
    """
    
    def __init__(self):
        self.baseline_memory = self._get_memory_usage()
        self.measurements = {}
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    @contextmanager
    def measure(self, operation_name: str):
        """Context manager for measuring memory usage."""
        start_memory = self._get_memory_usage()
        try:
            yield
        finally:
            end_memory = self._get_memory_usage()
            memory_delta = end_memory - start_memory
            
            if operation_name not in self.measurements:
                self.measurements[operation_name] = []
            self.measurements[operation_name].append(memory_delta)
    
    def get_stats(self, operation_name: str) -> Dict[str, float]:
        """Get memory statistics for an operation."""
        if operation_name not in self.measurements:
            return {}
        
        measurements = self.measurements[operation_name]
        return {
            "mean_memory_delta": np.mean(measurements),
            "std_memory_delta": np.std(measurements),
            "min_memory_delta": np.min(measurements),
            "max_memory_delta": np.max(measurements),
            "total_calls": len(measurements)
        }
    
    def get_current_usage(self) -> float:
        """Get current memory usage relative to baseline."""
        return self._get_memory_usage() - self.baseline_memory


class QuantumCircuitProfiler:
    """
    Profiler for quantum circuit performance metrics.
    """
    
    def __init__(self):
        self.circuit_metrics = {}
        
    def record_circuit_execution(
        self, 
        circuit_name: str,
        n_qubits: int,
        circuit_depth: int,
        gate_count: int,
        execution_time: float,
        backend_type: str = "simulator"
    ):
        """Record metrics for a quantum circuit execution."""
        if circuit_name not in self.circuit_metrics:
            self.circuit_metrics[circuit_name] = []
        
        self.circuit_metrics[circuit_name].append({
            "n_qubits": n_qubits,
            "circuit_depth": circuit_depth,
            "gate_count": gate_count,
            "execution_time": execution_time,
            "backend_type": backend_type
        })
    
    def get_circuit_stats(self, circuit_name: str) -> Dict[str, Any]:
        """Get statistics for a quantum circuit."""
        if circuit_name not in self.circuit_metrics:
            return {}
        
        metrics = self.circuit_metrics[circuit_name]
        
        # Extract numeric metrics
        execution_times = [m["execution_time"] for m in metrics]
        depths = [m["circuit_depth"] for m in metrics]
        gate_counts = [m["gate_count"] for m in metrics]
        
        return {
            "mean_execution_time": np.mean(execution_times),
            "std_execution_time": np.std(execution_times),
            "mean_circuit_depth": np.mean(depths),
            "mean_gate_count": np.mean(gate_counts),
            "total_executions": len(metrics),
            "backend_types": list(set(m["backend_type"] for m in metrics))
        }
    
    @contextmanager
    def measure_circuit(
        self, 
        circuit_name: str,
        n_qubits: int,
        circuit_depth: int = None,
        gate_count: int = None,
        backend_type: str = "simulator"
    ):
        """Context manager for measuring quantum circuit execution."""
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            
            self.record_circuit_execution(
                circuit_name, n_qubits, circuit_depth or 0, 
                gate_count or 0, execution_time, backend_type
            )


class ThroughputMeasurer:
    """
    Measurer for system throughput.
    """
    
    def __init__(self):
        self.measurements = {}
    
    def measure_throughput(
        self, 
        operation_name: str,
        operation_func: Callable,
        test_data: List[Any],
        duration: float = 60.0
    ) -> Dict[str, float]:
        """
        Measure throughput for an operation.
        
        Args:
            operation_name: Name of the operation
            operation_func: Function to call for each data item
            test_data: List of test data items
            duration: Duration to run the test (seconds)
            
        Returns:
            Throughput metrics
        """
        start_time = time.time()
        processed_count = 0
        errors = 0
        
        data_index = 0
        while time.time() - start_time < duration:
            try:
                # Cycle through test data
                if data_index >= len(test_data):
                    data_index = 0
                
                operation_func(test_data[data_index])
                processed_count += 1
                data_index += 1
                
            except Exception as e:
                errors += 1
                data_index += 1
        
        actual_duration = time.time() - start_time
        throughput = processed_count / actual_duration
        error_rate = errors / (processed_count + errors) if (processed_count + errors) > 0 else 0.0
        
        metrics = {
            "throughput_ops_per_sec": throughput,
            "total_operations": processed_count,
            "total_errors": errors,
            "error_rate": error_rate,
            "test_duration": actual_duration
        }
        
        self.measurements[operation_name] = metrics
        return metrics


def compute_efficiency_metrics(
    classical_results: Dict[str, float],
    quantum_results: Dict[str, float]
) -> Dict[str, Any]:
    """
    Compare efficiency between classical and quantum systems.
    
    Args:
        classical_results: Performance metrics for classical system
        quantum_results: Performance metrics for quantum system
        
    Returns:
        Efficiency comparison metrics
    """
    comparison = {}
    
    # Latency comparison
    if "mean_latency" in classical_results and "mean_latency" in quantum_results:
        classical_latency = classical_results["mean_latency"]
        quantum_latency = quantum_results["mean_latency"]
        
        comparison["latency_speedup"] = classical_latency / quantum_latency
        comparison["latency_overhead"] = quantum_latency - classical_latency
        comparison["latency_improvement"] = (classical_latency - quantum_latency) / classical_latency * 100
    
    # Memory comparison
    if "mean_memory_delta" in classical_results and "mean_memory_delta" in quantum_results:
        classical_memory = classical_results["mean_memory_delta"]
        quantum_memory = quantum_results["mean_memory_delta"]
        
        comparison["memory_ratio"] = quantum_memory / classical_memory if classical_memory != 0 else float('inf')
        comparison["memory_overhead"] = quantum_memory - classical_memory
    
    # Throughput comparison
    if "throughput_ops_per_sec" in classical_results and "throughput_ops_per_sec" in quantum_results:
        classical_throughput = classical_results["throughput_ops_per_sec"]
        quantum_throughput = quantum_results["throughput_ops_per_sec"]
        
        comparison["throughput_ratio"] = quantum_throughput / classical_throughput
        comparison["throughput_improvement"] = (quantum_throughput - classical_throughput) / classical_throughput * 100
    
    # Error rate comparison
    if "error_rate" in classical_results and "error_rate" in quantum_results:
        classical_error = classical_results["error_rate"]
        quantum_error = quantum_results["error_rate"]
        
        comparison["error_rate_difference"] = quantum_error - classical_error
        comparison["error_rate_ratio"] = quantum_error / classical_error if classical_error != 0 else float('inf')
    
    return comparison


def benchmark_rag_pipeline(
    pipeline_func: Callable,
    test_queries: List[str],
    system_name: str = "RAG System",
    duration: float = 300.0  # 5 minutes
) -> Dict[str, Any]:
    """
    Comprehensive benchmark of a RAG pipeline.
    
    Args:
        pipeline_func: Function that processes a query and returns an answer
        test_queries: List of test queries
        system_name: Name of the system being benchmarked
        duration: Duration to run the benchmark
        
    Returns:
        Comprehensive benchmark results
    """
    print(f"Benchmarking {system_name}...")
    
    # Initialize profilers
    latency_profiler = LatencyProfiler()
    memory_profiler = MemoryProfiler()
    throughput_measurer = ThroughputMeasurer()
    
    # Warm-up runs
    print("Warming up...")
    for i in range(min(3, len(test_queries))):
        try:
            pipeline_func(test_queries[i])
        except Exception as e:
            print(f"Warm-up error: {e}")
    
    # Latency measurements
    print("Measuring latency...")
    latencies = []
    for query in test_queries[:10]:  # Use first 10 queries for latency
        with latency_profiler.measure("pipeline"):
            with memory_profiler.measure("pipeline"):
                try:
                    start_time = time.time()
                    pipeline_func(query)
                    latencies.append(time.time() - start_time)
                except Exception as e:
                    print(f"Error processing query: {e}")
    
    # Throughput measurement
    print("Measuring throughput...")
    throughput_metrics = throughput_measurer.measure_throughput(
        "pipeline", pipeline_func, test_queries, duration=min(duration, 60.0)
    )
    
    # Compile results
    latency_stats = latency_profiler.get_stats("pipeline")
    memory_stats = memory_profiler.get_stats("pipeline")
    
    results = {
        "system_name": system_name,
        "benchmark_timestamp": time.time(),
        "latency_metrics": latency_stats,
        "memory_metrics": memory_stats,
        "throughput_metrics": throughput_metrics,
        "test_queries_count": len(test_queries),
        "successful_latency_measurements": len(latencies)
    }
    
    return results


def create_performance_report(
    benchmark_results: Dict[str, Any],
    comparison_results: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a formatted performance report.
    
    Args:
        benchmark_results: Results from benchmark_rag_pipeline
        comparison_results: Optional comparison with another system
        
    Returns:
        Formatted performance report
    """
    system_name = benchmark_results.get("system_name", "Unknown System")
    
    report = f"\n{'='*60}\n"
    report += f"Performance Benchmark Report: {system_name}\n"
    report += f"{'='*60}\n\n"
    
    # Latency metrics
    latency_metrics = benchmark_results.get("latency_metrics", {})
    if latency_metrics:
        report += "Latency Metrics:\n"
        report += "-" * 15 + "\n"
        report += f"  Mean Latency: {latency_metrics.get('mean_latency', 0):.4f}s\n"
        report += f"  Median Latency: {latency_metrics.get('median_latency', 0):.4f}s\n"
        report += f"  95th Percentile: {latency_metrics.get('p95_latency', 0):.4f}s\n"
        report += f"  99th Percentile: {latency_metrics.get('p99_latency', 0):.4f}s\n"
        report += f"  Standard Deviation: {latency_metrics.get('std_latency', 0):.4f}s\n\n"
    
    # Memory metrics
    memory_metrics = benchmark_results.get("memory_metrics", {})
    if memory_metrics:
        report += "Memory Metrics:\n"
        report += "-" * 14 + "\n"
        report += f"  Mean Memory Delta: {memory_metrics.get('mean_memory_delta', 0):.2f} MB\n"
        report += f"  Max Memory Delta: {memory_metrics.get('max_memory_delta', 0):.2f} MB\n\n"
    
    # Throughput metrics
    throughput_metrics = benchmark_results.get("throughput_metrics", {})
    if throughput_metrics:
        report += "Throughput Metrics:\n"
        report += "-" * 18 + "\n"
        report += f"  Throughput: {throughput_metrics.get('throughput_ops_per_sec', 0):.2f} ops/sec\n"
        report += f"  Total Operations: {throughput_metrics.get('total_operations', 0)}\n"
        report += f"  Error Rate: {throughput_metrics.get('error_rate', 0):.2%}\n\n"
    
    # Comparison section
    if comparison_results:
        report += "Comparison Results:\n"
        report += "-" * 18 + "\n"
        
        if "latency_speedup" in comparison_results:
            speedup = comparison_results["latency_speedup"]
            report += f"  Latency Speedup: {speedup:.2f}x\n"
        
        if "throughput_improvement" in comparison_results:
            improvement = comparison_results["throughput_improvement"]
            report += f"  Throughput Improvement: {improvement:.1f}%\n"
        
        if "memory_overhead" in comparison_results:
            overhead = comparison_results["memory_overhead"]
            report += f"  Memory Overhead: {overhead:.2f} MB\n"
        
        report += "\n"
    
    return report


def save_benchmark_results(
    results: Dict[str, Any], 
    filepath: str
):
    """
    Save benchmark results to a JSON file.
    
    Args:
        results: Benchmark results dictionary
        filepath: Path to save the results
    """
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def clean_dict(d):
        if isinstance(d, dict):
            return {k: clean_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [clean_dict(v) for v in d]
        else:
            return convert_numpy(d)
    
    clean_results = clean_dict(results)
    
    with open(filepath, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    print(f"Benchmark results saved to {filepath}")


def load_benchmark_results(filepath: str) -> Dict[str, Any]:
    """
    Load benchmark results from a JSON file.
    
    Args:
        filepath: Path to the results file
        
    Returns:
        Benchmark results dictionary
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    return results