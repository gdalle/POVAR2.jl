using Pkg
Pkg.activate(dirname(@__DIR__))
Pkg.instantiate()

using CairoMakie
using POVAR2
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

## Dense

plot_p(
    Random.seed!(rng, 63),
    joinpath(PLOTS_PATH, "p_dense.pdf");
    p_vals=10 .^ range(-1.5, 0, 20),
    T=10_000,
    D=5,
    S=5,
    σ=1.0,
    ω=0.1,
)
plot_T(
    Random.seed!(rng, 63),
    joinpath(PLOTS_PATH, "T_dense.pdf");
    T_vals=round.(Int, 10 .^ range(3, 5, 20)),
    D=5,
    S=5,
    σ=1.0,
    ω=0.1,
)
plot_D(
    Random.seed!(rng, 63),
    joinpath(PLOTS_PATH, "D_dense.pdf");
    D_vals=round.(Int, 10 .^ range(0, 2, 20)),
    T=10_000,
    σ=1.0,
    ω=0.1,
)
plot_ω(
    Random.seed!(rng, 63),
    joinpath(PLOTS_PATH, "omega_dense.pdf");
    ω_vals=10 .^ range(-2, 1, 20),
    T=10_000,
    D=5,
    S=5,
    σ=1.0,
)

## Sparse

plot_p(
    Random.seed!(rng, 63),
    joinpath(PLOTS_PATH, "p_sparse.pdf");
    p_vals=10 .^ range(-1.5, 0, 20),
    T=50_000,
    D=50,
    S=5,
    σ=1.0,
    ω=0.1,
)
plot_T(
    Random.seed!(rng, 63),
    joinpath(PLOTS_PATH, "T_sparse.pdf");
    T_vals=round.(Int, 10 .^ range(4, 6, 20)),
    D=50,
    S=5,
    σ=1.0,
    ω=0.1,
    legend_position=:lb,
)
plot_S(
    Random.seed!(rng, 63),
    joinpath(PLOTS_PATH, "S_sparse.pdf");
    S_vals=2:2:20,
    D=50,
    T=50_000,
    σ=1.0,
    ω=0.1,
)
plot_ω(
    Random.seed!(rng, 63),
    joinpath(PLOTS_PATH, "omega_sparse.pdf");
    ω_vals=10 .^ range(-2, 1, 20),
    T=50_000,
    D=50,
    S=5,
    σ=1.0,
)

plot_SD(
    Random.seed!(rng, 63),
    joinpath(PLOTS_PATH, "SD_sparse.pdf");
    S_vals=2:3:20,
    D_vals=20:5:50,
    T=50_000,
    p=1.0,
    σ=1.0,
    ω=0.1,
)
