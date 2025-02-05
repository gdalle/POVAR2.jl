struct POVARModel{R<:Real,M<:AbstractMatrix{R}}
    θ::M
    p::R
    σ::R
    ω::R
    T::Int

    function POVARModel(; θ::M, p::R, σ::R, ω::R, T::Int) where {R,M}
        # correct values
        @assert opnorm(θ, 2) < 1
        @assert p >= zero(R)
        @assert σ >= zero(R)
        @assert ω >= zero(R)
        # correct shapes
        checksquare(θ)
        # construct
        return new{R,M}(θ, p, σ, ω, T)
    end
end

dimension(model::POVARModel) = size(model.θ, 1)
duration(model::POVARModel) = model.T

@kwdef struct Dataset{XT<:AbstractMatrix,PT<:AbstractMatrix,YT<:AbstractMatrix}
    X::XT
    proj::PT
    Y::YT
end

duration(dataset::Dataset) = size(dataset.X, 2)

function split(dataset::Dataset)
    (; X, proj, Y) = dataset
    T_split = duration(dataset) ÷ 2
    X1, X2 = X[:, 1:T_split], X[:, (T_split + 1):end]
    proj1, proj2 = proj[:, 1:T_split], proj[:, (T_split + 1):end]
    Y1, Y2 = Y[:, 1:T_split], Y[:, (T_split + 1):end]
    d1 = Dataset(X1, proj1, Y1)
    d2 = Dataset(X2, proj2, Y2)
    return d1, d2
end

function Base.rand(rng::AbstractRNG, model::POVARModel{R}, K::Integer=1) where {R}
    (; θ, p, σ, ω, T) = model
    D = size(θ, 1)

    ε = rand(rng, Normal(0, σ), D, T, K)
    η = rand(rng, Normal(0, ω), D, T, K)
    proj = rand(rng, Bernoulli(p), D, T, K)

    X = Array{R,3}(undef, D, T, K)
    @views for k in 1:K
        X[:, 1, k] .= ε[:, 1, k]  # not stationary
        for t in 2:T
            mul!(X[:, t, k], θ, X[:, t - 1, k])
            X[:, t, k] .+= ε[:, t, k]
        end
    end
    Y = proj .* X .+ η

    return @views [Dataset(; X=X[:, :, k], proj=proj[:, :, k], Y=Y[:, :, k]) for k in 1:K]
end
