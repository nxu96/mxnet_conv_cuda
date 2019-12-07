# Introduction

* Get practical experience by using, profiling, and modifying MXNet, a standard open-source neural-network framework.
* Demonstrate command of CUDA and optimization approaches by designing and implementing an optimized neural-network convolution layer forward pass.

# Optimizations

- [ ] Shared Memory convolution
- [ ] Weight matrix (kernel values) in constant memory (Feasible ? TBD)
- [x] Loop unrolling 
- [x] Unroll + shared-memory Matrix multiply
- [ ] Kernel fusion for unrolling and matrix-multiplication
- [ ] Exploiting parallelism in input images, input channels, and output channels
- [x] Multiple kernel implementations for different layer sizes
- [ ] Sweeping various parameters to find best values (block sizes, amount of thread coarsening)

