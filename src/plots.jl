DEFAULT_SAMPLES = 10
p_vals = reverse([1, 1 / 2, 1 / 4, 1 / 8])

function add_band!(ax, x, yp; label=nothing, slope=false, kwargs...)
    y = pmedian.(yp)
    ylow = pquantile.(yp, 0.1)
    yhigh = pquantile.(yp, 0.9)
    scatter!(ax, x, y; markersize=10, label=label, kwargs...)
    band!(ax, x, ylow, yhigh; alpha=0.5, kwargs...)
    if slope
        α, β = round.(theil_sen(log10.(x), log10.(y)), digits=2)
        lines!(ax, x, 10 .^ (α * log10.(x) .+ β); label=L"\text{slope} = %$α", kwargs...)
    end
end

function plot_p(
    rng, path=nothing, samples=DEFAULT_SAMPLES; p_vals, T, D, S, σ, ω, legend_position=:rt
)
    estimator = D == S ? DenseEstimator() : SparseEstimator()
    p_errors = @showprogress map(p_vals) do p
        estimation_error_particles(rng, estimator; T, D, S, p, σ, ω, samples)
    end
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        title=if estimator isa DenseEstimator
            L"Dense estimator - $T = %$T, ~D = %$D, ~s = %$S, ~\sigma = %$σ, ~\omega = %$ω$"
        else
            L"Sparse estimator - $T = %$T, ~D = %$D, ~s = %$S, ~\sigma = %$σ, ~\omega = %$ω$"
        end,
        xlabel=L"Sampling probability $p$",
        ylabel=L"Estimation error $||\hat{\theta} - \theta ||_{\infty}$",
        yscale=log10,
        xscale=log10,
        aspect=1,
    )
    x, yp = p_vals, p_errors
    add_band!(ax, x, yp; label=nothing, slope=true, color=:black)
    axislegend(ax; nbanks=2, position=legend_position)
    !isnothing(path) && save(path, fig)
    return fig
end

function plot_T(
    rng, path=nothing, samples=DEFAULT_SAMPLES; T_vals, D, S, σ, ω, legend_position=:rt
)
    estimator = D == S ? DenseEstimator() : SparseEstimator()
    T_errors = map(p_vals) do p
        @showprogress "p = $p" map(T_vals) do T
            estimation_error_particles(rng, estimator; T, D, S, p, σ, ω, samples)
        end
    end
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        title=if estimator isa DenseEstimator
            L"Dense estimator - $D = %$D, ~s = %$S, ~\sigma = %$σ, ~\omega = %$ω$"
        else
            L"Sparse estimator - $D = %$D, ~s = %$S, ~\sigma = %$σ, ~\omega = %$ω$"
        end,
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
    axislegend(ax; nbanks=2, position=legend_position)
    !isnothing(path) && save(path, fig)
    return fig
end

function plot_D(
    rng, path=nothing, samples=DEFAULT_SAMPLES; D_vals, T, σ, ω, legend_position=:lt
)
    estimator = DenseEstimator()
    D_errors = map(p_vals) do p
        @showprogress "p = $p" map(D_vals) do D
            estimation_error_particles(rng, estimator; D, S=D, p, σ, ω, T, samples)
        end
    end
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        title=L"Dense estimator - $T = %$T, ~s = D, ~\sigma = %$σ, ~\omega = %$ω$",
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
    axislegend(ax; nbanks=2, position=legend_position)
    !isnothing(path) && save(path, fig)
    return fig
end

function plot_S(
    rng, path=nothing, samples=DEFAULT_SAMPLES; S_vals, T, D, σ, ω, legend_position=:lt
)
    estimator = SparseEstimator()
    S_errors = map(p_vals) do p
        @showprogress "p = $p" map(S_vals) do S
            estimation_error_particles(rng, estimator; T, D, S, p, σ, ω, samples)
        end
    end
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        title=L"Sparse estimator - $T = %$T, ~D = %$D, ~\sigma = %$σ, ~\omega = %$ω$",
        xlabel=L"Sparsity level $s$",
        ylabel=L"Estimation error $||\hat{\theta} - \theta ||_{\infty}$",
        yscale=log10,
        xscale=log10,
        aspect=1,
    )
    for (k, p) in enumerate(p_vals)
        x, yp = S_vals, S_errors[k]
        add_band!(ax, x, yp; label=L"p=%$p", slope=true)
    end
    axislegend(ax; nbanks=2, position=legend_position)
    !isnothing(path) && save(path, fig)
    return fig
end

function plot_ω(
    rng, path=nothing, samples=DEFAULT_SAMPLES; ω_vals, T, D, S, σ, legend_position=:lt
)
    estimator = D == S ? DenseEstimator() : SparseEstimator()
    ω_errors = map(p_vals) do p
        @showprogress "p = $p" map(ω_vals) do ω
            estimation_error_particles(rng, estimator; T, D, S, p, σ, ω, samples)
        end
    end
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        title=if estimator isa DenseEstimator
            L"Dense estimator - $T = %$T, ~D = %$D, ~s = %$S, ~\sigma = %$σ$"
        else
            L"Sparse estimator - $T = %$T, ~D = %$D, ~s = %$S, ~\sigma = %$σ$"
        end,
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
    axislegend(ax; nbanks=2, position=legend_position)
    !isnothing(path) && save(path, fig)
    return fig
end

function plot_SD(rng, path=nothing, samples=DEFAULT_SAMPLES; D_vals, S_vals, p, T, σ, ω)
    estimator = SparseEstimator()
    SD_vals = collect(Iterators.product(S_vals, D_vals))
    SD_errors = similar(SD_vals, Float64)
    @showprogress for i in eachindex(SD_vals, SD_errors)
        S, D = SD_vals[i]
        SD_errors[i] = pmedian(
            estimation_error_particles(rng, estimator; T, D, S, p, σ, ω, samples)
        )
    end

    fig = Figure()
    ax = Axis(
        fig[1, 1];
        title=L"Sparse estimator - $T = %$T, ~p = %$p, ~\sigma = %$σ, ~\omega = %$ω$",
        xlabel=L"Sparsity level $s$",
        ylabel=L"State dimension $D$",
        aspect=1,
    )
    h = heatmap!(ax, S_vals, D_vals, SD_errors)
    Colorbar(
        fig[:, end + 1], h; label=L"Estimation error $||\hat{\theta} - \theta ||_{\infty}$"
    )
    !isnothing(path) && save(path, fig)
    return fig
end
