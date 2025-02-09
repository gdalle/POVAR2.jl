DEFAULT_SAMPLES = 10
p_vals = reverse([1, 1 / 2, 1 / 4, 1 / 8])

function add_band!(ax, x, yp; label=nothing, slope=false)
    y = pmedian.(yp)
    ylow = pquantile.(yp, 0.1)
    yhigh = pquantile.(yp, 0.9)
    scatter!(ax, x, y; markersize=10, label=label)
    band!(ax, x, ylow, yhigh; alpha=0.5)
    if slope
        α, β = round.(theil_sen(log10.(x), log10.(y)), digits=2)
        lines!(ax, x, 10 .^ (α * log10.(x) .+ β); label=L"\text{slope} = %$α")
    end
end

function plot_T(rng, path=nothing, samples=DEFAULT_SAMPLES; T_vals, D, S, σ, ω)
    estimator = D == S ? DenseEstimator() : SparseEstimator()
    T_errors = map(p_vals) do p
        @showprogress "p = $p" map(T_vals) do T
            estimation_error_particles(rng, estimator; D, S, p, σ, ω, T, samples)
        end
    end
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        title=L"D = %$D, ~S = %$S, ~\sigma = %$σ, ~\omega = %$ω",
        xlabel=L"Period length $T$",
        ylabel=L"Estimation error $||\hat{\theta} - \theta ||_{\infty}$",
        yscale=log10,
        xscale=log10,
        aspect=1,
    )
    for (k, p) in enumerate(p_vals)
        x, yp = T_vals, T_errors[k]
        add_band!(ax, x, yp; label=L"p=%$p", slope=true)
    end
    axislegend(ax; nbanks=2, position=:rt)
    !isnothing(path) && save(path, fig)
    return fig
end

function plot_D(rng, path=nothing, samples=DEFAULT_SAMPLES; D_vals, T, σ, ω)
    estimator = DenseEstimator()
    D_errors = map(p_vals) do p
        @showprogress "p = $p" map(D_vals) do D
            estimation_error_particles(rng, estimator; D, S=D, p, σ, ω, T, samples)
        end
    end
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        title=L"T = %$T, ~S = D, ~\sigma = %$σ, ~\omega = %$ω",
        xlabel=L"State dimension $D$",
        ylabel=L"Estimation error $||\hat{\theta} - \theta ||_{\infty}$",
        yscale=log10,
        xscale=log10,
        aspect=1,
    )
    for (k, p) in enumerate(p_vals)
        x, yp = D_vals, D_errors[k]
        add_band!(ax, x, yp; label=L"p=%$p", slope=true)
    end
    axislegend(ax; nbanks=2, position=:lt)
    !isnothing(path) && save(path, fig)
    return fig
end

function plot_ω(rng, path=nothing, samples=DEFAULT_SAMPLES; ω_vals, T, D, S, σ)
    estimator = D == S ? DenseEstimator() : SparseEstimator()
    ω_errors = map(p_vals) do p
        @showprogress "p = $p" map(ω_vals) do ω
            estimation_error_particles(rng, estimator; D, S, p, σ, ω, T, samples)
        end
    end
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        title=L"T = %$T, ~D = %$D, ~S = %$S, ~\sigma = %$σ",
        xlabel=L"Noise standard deviation $\omega$",
        ylabel=L"Estimation error $||\hat{\theta} - \theta ||_{\infty}$",
        yscale=log10,
        xscale=log10,
        aspect=1,
    )
    for (k, p) in enumerate(p_vals)
        x, yp = ω_vals, ω_errors[k]
        add_band!(ax, x, yp; label=L"p=%$p", slope=false)
    end
    axislegend(ax; nbanks=2, position=:lt)
    !isnothing(path) && save(path, fig)
    return fig
end
