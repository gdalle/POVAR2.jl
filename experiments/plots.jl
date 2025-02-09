using Pkg
Pkg.activate(dirname(@__DIR__))
# Pkg.instantiate()

using CairoMakie
using MonteCarloMeasurements
using POVAR2
using ProgressMeter
using Random

MT = Makie.MathTeXEngine
mt_fonts_dir = joinpath(dirname(pathof(MT)), "..", "assets", "fonts", "NewComputerModern")

update_theme!(;
    Scatter=(; cycle=Cycle([:color, :marker, :linestyle]; covary=true)),
    Lines=(; cycle=Cycle([:color, :linestyle]; covary=true)),
    fonts=(
        regular=joinpath(mt_fonts_dir, "NewCM10-Regular.otf"),
        bold=joinpath(mt_fonts_dir, "NewCM10-Bold.otf"),
    ),
)

rng = Random.default_rng()

PLOTS_PATH = joinpath(@__DIR__, "img/")

plot_T(
    Random.seed!(rng, 63);
    T_vals=round.(Int, 10 .^ range(3, 5, 20)),
    D=10,
    S=10,
    σ=1.0,
    ω=0.1,
)
plot_D(
    Random.seed!(rng, 63); D_vals=round.(Int, 10 .^ range(0, 2, 20)), T=10_000, σ=1.0, ω=0.1
)
plot_ω(Random.seed!(rng, 63); ω_vals=10 .^ range(-2, 2, 20), T=10_000, D=10, S=10, σ=1.0)

plot_T(
    Random.seed!(rng, 63);
    T_vals=round.(Int, 10 .^ range(3, 5, 20)),
    D=50,
    S=2,
    σ=1.0,
    ω=0.1,
)
plot_ω(Random.seed!(rng, 63); ω_vals=10 .^ range(-2, 2, 20), T=10_000, D=50, S=2, σ=1.0)
