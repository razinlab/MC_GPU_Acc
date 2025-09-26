# GPU-Accelerated Options Pricing with Monte Carlo Simulation and Deep Learning Surrogate

A high-performance options pricing system that combines GPU-accelerated Monte Carlo simulation with deep learning surrogate modeling to achieve both accuracy and ultra-fast inference times.

## Key Features

• Custom CUDA Monte Carlo simulator with >600,000x speedup over baseline Python implementation
• Deep neural network surrogate model trained on 1.5M simulation samples  
• Sub-millisecond inference with ≤1% MAE error on key test cases
• PyBind11 integration for seamless Python-CUDA interoperability
• Comprehensive validation against Black-Scholes analytical solutions

## Methodology

The project implements a multi-stage approach to options pricing optimization:

**Stage 1: Baseline Implementation**
Started with traditional Black-Scholes analytical pricing and pure Python Monte Carlo simulation to establish accuracy benchmarks and performance baselines.

**Stage 2: GPU Acceleration** 
Developed a custom CUDA kernel for Monte Carlo path simulation, leveraging cuRAND for high-quality random number generation and optimized memory access patterns for maximum throughput.

**Stage 3: Surrogate Modeling**
Generated 1.5M training samples across diverse market conditions using stratified sampling buckets (micro-cap to mega-cap scenarios) and trained a deep neural network to approximate the simulation results with sub-millisecond inference.

**Stage 4: Integration & Validation**
Created PyBind11 bindings for seamless Python integration and validated the complete pipeline against analytical solutions across various option scenarios.

## Performance Results

**Simulation Throughput:**
- Python CPU: ~1,212 simulations/second
- CUDA GPU: ~748M simulations/second  
- **Speedup: 600,000x improvement**

**Inference Performance:**
- GPU Monte Carlo: ~100ms for 262K simulations
- Neural Network: <1ms inference time
- **Accuracy: ≤1% MAE on validated test cases**

## Architecture

The system architecture consists of three main components:

1. **CUDA Simulation Engine**: Custom kernel implementing geometric Brownian motion with optimized memory coalescing
2. **Deep Learning Surrogate**: 3-layer neural network (256 hidden units) with ReLU activation and Adam optimization
3. **PyBind11 Interface**: Seamless Python-CUDA integration layer for production deployment

## Findings

The GPU-accelerated Monte Carlo simulator achieved exceptional performance scaling, enabling the generation of large-scale training datasets that would be computationally prohibitive with traditional CPU approaches. The neural network surrogate successfully learned the complex option pricing function across diverse market regimes, with particularly strong performance on at-the-money and high-volatility scenarios.

Performance analysis revealed that while the neural network occasionally struggled with very short expiry options (20% error on 1-month contracts), it maintained excellent accuracy (0.5-5% error) across the majority of test scenarios, making it suitable for production applications requiring ultra-fast pricing.

## Technical Implementation

**Dependencies:**
- CUDA 11.8+ with compute capability 8.9+
- PyTorch for neural network training
- PyBind11 for Python-CUDA integration  
- NumPy, SciPy for mathematical operations
- scikit-learn for data preprocessing

**Key optimizations:**
- Coalesced memory access patterns in CUDA kernels
- Stratified sampling across market regimes
- StandardScaler preprocessing for neural network stability
- Batch processing for efficient GPU utilization
