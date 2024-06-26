struct POVARModel{R<:Real,M1<:AbstractMatrix{R},M2<:AbstractMatrix{R},V<:AbstractVector{R}}
    θ::M1
    Σ::M2
    p::V
    ω::R

    function POVARModel(; θ::M1, Σ::M2, p::V, ω::R) where {R,M1,M2,V}
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
        return new{R,M1,M2,V}(θ, Σ, p, ω)
    end
end

function Base.rand(rng::AbstractRNG, m::POVARModel{R}, T::Integer) where {R}
    (; θ, Σ, p, ω) = m
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

    π = Matrix{Bool}(undef, D, T)
    @views for d in 1:D
        rand!(Bernoulli(p[d]), π[d, :])
    end

    Y = π .* X .+ η

    return (; X, π, Y)
end
