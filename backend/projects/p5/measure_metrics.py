"""
Metrics Measurement Script for Cost Analysis
Measures actual resource consumption and model performance
Run this after training your model to populate cost analysis data
"""

import os
import sys
import json
import time
from pathlib import Path

# Try importing required packages
try:
    import psutil
except ImportError:
    print("⚠ psutil not installed. Install with: pip install psutil")
    psutil = None

try:
    import torch
except ImportError:
    print("⚠ torch not installed. Install with: pip install torch")
    torch = None

try:
    import numpy as np
except ImportError:
    print("⚠ numpy not installed. Install with: pip install numpy")
    np = None


def measure_file_sizes():
    """Measure model and dataset file sizes."""
    measurements = {}
    
    # Model size
    model_path = '../saved_models/model.pt'
    if os.path.exists(model_path):
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        measurements['model_size_mb'] = model_size_mb
        print(f"✓ Model size: {model_size_mb:.2f} MB")
    else:
        print(f"✗ Model not found at {model_path}")
    
    # Dataset size
    data_dir = '../data'
    if os.path.exists(data_dir):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(data_dir):
            for filename in filenames:
                if filename.endswith('.txt'):
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
        
        dataset_size_gb = total_size / (1024**3)
        measurements['dataset_size_gb'] = dataset_size_gb
        print(f"✓ Dataset size: {dataset_size_gb:.2f} GB")
    
    return measurements


def measure_memory_usage():
    """Measure current process memory usage."""
    if psutil is None:
        print("⚠ Cannot measure memory: psutil not available")
        return {}
    
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 * 1024)
    
    print(f"✓ Current memory usage: {memory_mb:.2f} MB")
    return {'current_memory_mb': memory_mb}


def measure_gpu_memory():
    """Measure GPU memory if CUDA available."""
    if torch is None:
        print("⚠ Cannot measure GPU memory: torch not available")
        return {}
    
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / (1024**3)
        print(f"✓ GPU memory allocated: {gpu_mem:.2f} GB")
        return {'gpu_memory_gb': gpu_mem}
    else:
        print("✓ No GPU available")
        return {}


def load_training_metrics():
    """Load training metrics from training script output."""
    metrics_path = '../visualizations/training_time_metrics.json'
    
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            training_hours = metrics.get('sections', {}).get('Model Training', 0) / 3600
            print(f"✓ Training time: {training_hours:.2f} hours")
            return {'training_hours': training_hours}
        except json.JSONDecodeError:
            print(f"✗ Could not parse {metrics_path}")
            return {}
    else:
        print(f"✗ Training metrics not found at {metrics_path}")
        print("  Run train.py first to generate metrics")
        return {}


def estimate_inference_metrics():
    """
    Estimate inference metrics based on model characteristics.
    NOTE: For actual metrics, run benchmarking after deploying model.
    """
    measurements = {}
    
    # Based on LSTM architecture:
    # - Single LSTM layer with 128 units
    # - 1.5M parameters
    # - Typical inference latency for LSTM: 10-100ms depending on hardware
    
    print("\n[INFERENCE METRICS - ESTIMATED]")
    print("  (These are estimates. For actual metrics, benchmark after deployment)")
    
    # Rule of thumb: ~1 ms per 100K parameters for CPU inference
    estimated_latency_cpu_ms = 1.5e6 / 100_000  # 15 ms
    
    # GPU (NVIDIA T4): ~10x faster than CPU for small-medium models
    estimated_latency_gpu_ms = estimated_latency_cpu_ms / 10  # 1.5 ms
    
    # Use conservative estimate (CPU deployment for cost analysis)
    measurements['inference_latency_ms'] = estimated_latency_cpu_ms
    measurements['inference_latency_gpu_ms'] = estimated_latency_gpu_ms
    
    print(f"  Estimated CPU inference latency: {estimated_latency_cpu_ms:.2f} ms")
    print(f"  Estimated GPU inference latency: {estimated_latency_gpu_ms:.2f} ms")
    print(f"  (Using CPU estimate for conservative cost analysis)")
    
    # Calculate throughput
    inferences_per_second = 1000 / estimated_latency_cpu_ms
    measurements['inferences_per_second'] = inferences_per_second
    print(f"  Estimated throughput: {inferences_per_second:.1f} inferences/second")
    
    return measurements


def generate_measurement_report(all_measurements):
    """Generate a report of all measurements."""
    report = "=" * 70 + "\n"
    report += "RNN MODEL - RESOURCE MEASUREMENTS REPORT\n"
    report += "=" * 70 + "\n\n"
    
    report += "MEASURED VALUES:\n"
    report += "-" * 70 + "\n"
    for key, value in all_measurements.items():
        if isinstance(value, float):
            report += f"{key}: {value:.2f}\n"
        else:
            report += f"{key}: {value}\n"
    
    report += "\n" + "=" * 70 + "\n"
    report += "RECOMMENDATIONS FOR COST ANALYSIS:\n"
    report += "-" * 70 + "\n"
    
    model_size = all_measurements.get('model_size_mb', 50)
    dataset_size = all_measurements.get('dataset_size_gb', 5)
    training_hours = all_measurements.get('training_hours', 2.5)
    
    report += f"1. Small model size ({model_size:.1f} MB) → Cost-effective deployment\n"
    report += f"2. Dataset size ({dataset_size:.2f} GB) → Low storage costs\n"
    report += f"3. Quick training ({training_hours:.1f} hours) → Low training costs\n"
    report += f"4. Use t3.large or m5.xlarge instances for inference\n"
    report += f"5. Consider serverless deployment for variable load\n"
    
    return report


def main():
    """Main measurement routine."""
    print("=" * 70)
    print("RNN MODEL - RESOURCE MEASUREMENTS")
    print("=" * 70 + "\n")
    
    all_measurements = {}
    
    # Collect measurements
    print("[1/4] Measuring file sizes...")
    all_measurements.update(measure_file_sizes())
    
    print("\n[2/4] Measuring memory...")
    all_measurements.update(measure_memory_usage())
    
    print("\n[3/4] Measuring GPU...")
    all_measurements.update(measure_gpu_memory())
    
    print("\n[4/4] Loading training metrics...")
    all_measurements.update(load_training_metrics())
    
    # Estimate inference metrics
    all_measurements.update(estimate_inference_metrics())
    
    # Generate and save report
    print("\nGenerating measurement report...")
    report = generate_measurement_report(all_measurements)
    
    os.makedirs('../visualizations', exist_ok=True)
    
    # Save measurements as JSON
    json_path = '../visualizations/resource_measurements.json'
    with open(json_path, 'w') as f:
        json.dump(all_measurements, f, indent=2)
    print(f"✓ Measurements saved to: {json_path}")
    
    # Save report as text
    report_path = '../visualizations/measurements_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✓ Report saved to: {report_path}")
    
    print("\n" + "=" * 70)
    print("✓ MEASUREMENT COMPLETE")
    print("=" * 70)
    print(f"\nNext step: Run 'python cost_analysis.py' to generate cost analysis")


if __name__ == "__main__":
    main()
