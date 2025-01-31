using CairoMakie
using OhMyThreads
using POVAR2
using StableRNGs

rng = StableRNG(63)

PARAMS = (; D=5, σ=0.1, ω=0.1, p=1.0, T=10_000)

## Influence of T

p_vals = [0.1, 0.2, 0.5, 1.0]
T_vals = round.(Int, 10 .^ range(2, 4, 10))
T_errors = tmap(p_vals) do p
    rng = StableRNG(63)
    T_errors[p] = map(T_vals) do T
        model = random_povar(rng; p, T, D=PARAMS.D, S=PARAMS.D, σ=PARAMS.σ, ω=PARAMS.ω)
        estimation_error(rng, SparseEstimator(), model)
    end
end
