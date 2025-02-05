using Pkg
Pkg.activate(dirname(@__DIR__))
# Pkg.instantiate()

using CairoMakie
using MonteCarloMeasurements
using OhMyThreads
using Pkg
using POVAR2
using ProgressMeter
using StableRNGs

update_theme!(;
    Scatter=(; cycle=Cycle([:color, :marker, :linestyle]; covary=true)),
    Lines=(; cycle=Cycle([:color, :linestyle]; covary=true)),
)

PLOTS_PATH = joinpath(@__DIR__, "img/")

p_vals = reverse([1, 1 / 2, 1 / 4, 1 / 8])
DEFAULT_PARAMS = (; D=5, S=5, σ=1.0, ω=0.1, p=1.0, T=10_000)

function plot_T(rng, DEFAULT_PARAMS; samples=10)
    (; D, S, σ, ω) = DEFAULT_PARAMS
    T_vals = round.(Int, 10 .^ range(2, 5, 20))
    T_errors = map(p_vals) do p
        @showprogress map(T_vals) do T
            estimation_error_particles(rng, DenseEstimator(); D, S, p, σ, ω, T, samples)
        end
    end
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        xlabel=L"Period length $T$",
        ylabel=L"Estimation error $||\hat{\theta} - \theta ||_{\infty}$",
        yscale=log10,
        xscale=log10,
        aspect=1,
    )
    for (k, p) in enumerate(p_vals)
        x, yp = T_vals, T_errors[k]
        y = pmedian.(yp)
        ylow = pquantile.(yp, 0.1)
        yhigh = pquantile.(yp, 0.9)
        α, β = round.(theil_sen(log10.(x), log10.(y)), digits=2)
        scatter!(ax, x, y; markersize=10, label=L"p=%$p")
        lines!(ax, x, 10 .^ (α * log10.(x) .+ β); label=L"\text{slope:} %$α")
        band!(ax, x, ylow, yhigh; alpha=0.5)
    end
    axislegend(ax; nbanks=2, position=:rt)
    return fig
end

function plot_D(rng, DEFAULT_PARAMS; samples=10)
    (; T, σ, ω) = DEFAULT_PARAMS
    D_vals = round.(Int, 10 .^ range(0, 2, 20))
    D_errors = map(p_vals) do p
        @showprogress map(D_vals) do D
            estimation_error_particles(rng, DenseEstimator(); D, S, p, σ, ω, T, samples)
        end
    end
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        xlabel=L"State dimension $D$",
        ylabel=L"Estimation error $||\hat{\theta} - \theta ||_{\infty}$",
        yscale=log10,
        xscale=log10,
        aspect=1,
    )
    for (k, p) in enumerate(p_vals)
        x, yp = D_vals, D_errors[k]
        y = pmedian.(yp)
        ylow, yhigh = pquantile.(yp, 0.1), pquantile.(yp, 0.9)
        α, β = round.(theil_sen(log10.(x), log10.(y)), digits=2)
        scatter!(ax, x, y; markersize=10, label=L"p=%$p")
        lines!(ax, x, 10 .^ (α * log10.(x) .+ β); label=L"\text{slope:} %$α")
        band!(x, ylow, yhigh; alpha=0.5)
    end
    axislegend(ax; nbanks=2, position=:rb)
    return fig
end

function plot_ω(rng, DEFAULT_PARAMS; samples=10)
    (; T, D, σ) = DEFAULT_PARAMS
    ω_vals = 10 .^ range(-2, 2, 20)
    ω_errors = map(p_vals) do p
        @showprogress map(ω_vals) do ω
            estimation_error_particles(rng, DenseEstimator(); D, S, p, σ, ω, T, samples)
        end
    end
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        xlabel=L"Variance ratio $\omega^2 / \sigma^2$",
        ylabel=L"Estimation error $||\hat{\theta} - \theta ||_{\infty}$",
        yscale=log10,
        xscale=log10,
        aspect=1,
    )
    for (k, p) in enumerate(p_vals)
        x, yp = ω_vals .^ 2, ω_errors[k]
        y = pmedian.(yp)
        ylow, yhigh = pquantile.(yp, 0.1), pquantile.(yp, 0.9)
        α, β = round.(theil_sen(log10.(x), log10.(y)), digits=2)
        scatter!(ax, x, y; markersize=10, label=L"p=%$p")
        # lines!(ax, x, 10 .^ (α * log10.(x) .+ β); label=L"\text{slope:} %$α")
        band!(x, ylow, yhigh; alpha=0.5)
    end
    axislegend(ax; nbanks=2, position=:lt)
    return fig
end

plot_T(StableRNG(63), DEFAULT_PARAMS)
plot_D(StableRNG(63), DEFAULT_PARAMS)
plot_ω(StableRNG(63), DEFAULT_PARAMS)

(; σ, ω) = DEFAULT_PARAMS
T = 1000
p = 0.5
D_vals = 10:5:50
S_vals = 1:2:10

DS_errors = @showprogress map(Iterators.product(D_vals, S_vals)) do (D, S)
    yield()
    if D >= S
        return estimation_error_particles(StableRNG(63), SparseEstimator(); D, S, p, σ, ω, T, samples=30)
    else
        return Particles([NaN])
    end
end

contourf(D_vals, S_vals, pmean.(DS_errors); colormap=:plasma, levels=20)
