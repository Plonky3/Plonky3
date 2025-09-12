# Goldilocks WebAssembly Benchmark

This minimal WebAssembly module benchmarks field operations using native Goldilocks implementations:

- **Standard Goldilocks**: Direct field arithmetic implementation  
- **Goldilocks-Monty**: Montgomery arithmetic optimization

## Operations Benchmarked

Each benchmark performs 1000 operations and reports the execution time:

1. **Multiplication**: Field element multiplication
2. **Addition**: Field element addition

## Building

Make sure you have `wasm-pack` installed:

```bash
cargo install wasm-pack
```

Build the WebAssembly module:

```bash
wasm-pack build --target web --out-dir pkg
```

## Running

Serve the directory with a local HTTP server:

```bash
# Using Python
python3 -m http.server 8000

# Using Node.js
npx http-server

# Using Rust
cargo install basic-http-server
basic-http-server
```

Then open `http://localhost:8000` in your browser.


## Performance Comparison: WASM vs Native

| Operation | WASM (ms) | Native (ms) | WASM/Native Ratio | WASM Overhead |
|-----------|-----------|-------------|-------------------|---------------|
| **Goldilocks Addition** | 0.900 | 0.422 | 2.13x | +113% |
| **Goldilocks-Monty Addition** | 0.800 | 0.427 | 1.87x | +87% |
| **Goldilocks Multiplication** | 3.700 | 2.135 | 1.73x | +73% |
| **Goldilocks-Monty Multiplication** | 4.000 | 2.116 | 1.89x | +89% |
