# Zinc+

WIP implementation and playground on Nethermind's improvement to Zinc SNARK .

## License

Apache 2.0

## Would like to contribute?

see [Contributing](./CONTRIBUTING.md).




    
# Benches:

cargo bench --bench steps_sha256_8x_ecdsa_folded -p zinc-snark --features "parallel simd asm qx-constraints no-f2x true-ideal"

cargo bench --bench steps_sha256_8x_folded -p zinc-snark --features "parallel simd asm qx-constraints no-f2x true-ideal"

cargo bench --bench steps_big_linear_pi_folded -p zinc-snark --features "parallel simd asm"