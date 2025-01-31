struct POVARModel{R<:Real,M1<:AbstractMatrix{R},M2<:AbstractMatrix{R},V<:AbstractVector{R}}
    θ::M1
    Σ::M2
    p::V
    ω::R
    T::Int

    function POVARModel(; θ::M1, Σ::M2, p::V, ω::R, T::Int) where {R,M1,M2,V}
        # correct values
        @assert opnorm(θ, 2) < 1
        @assert issymmetric(Σ)
        @assert eigmin(Σ) >= zero(R)
        @assert all(<=(one(R)), p)
        @assert all(>=(zero(R)), p)
        @assert ω >= zero(R)
        # correct shapes
        D = checksquare(θ)
        @assert checksquare(Σ) == D
        @assert length(p) == D
        # construct
        return new{R,M1,M2,V}(θ, Σ, p, ω, T)
    end
end

@kwdef struct Dataset{XT,PT,YT}
    X::XT
    proj::PT
    Y::YT
end

function Base.rand(rng::AbstractRNG, model::POVARModel{R}) where {R}
    (; θ, Σ, p, ω, T) = model
    D = size(θ, 1)

    innovation_dist = MvNormal(zeros(D), Σ)
    ε = rand(rng, innovation_dist, T)

    noise_dist = MvNormal(zeros(D), Diagonal(fill(ω^2, D)))
    η = rand(rng, noise_dist, T)

    X = Matrix{R}(undef, D, T)
    X[:, 1] .= @view ε[:, 1]  # not stationary
    @views for t in 2:T
        mul!(X[:, t], θ, X[:, t - 1])
        X[:, t] .+= ε[:, t]
    end

    proj = Matrix{Bool}(undef, D, T)
    @views for d in 1:D
        rand!(rng, Bernoulli(p[d]), proj[d, :])
    end

    Y = proj .* X .+ η

    return Dataset(; X, proj, Y)
end
