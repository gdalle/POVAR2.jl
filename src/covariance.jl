function scaling(model::POVARModel, h::Integer)
    D = dimension(model)
    P = fill(model.p^2, D, D)
    if h == 0
        for d in axes(P, 1)
            P[d, d] = model.p
        end 
    end
    return P
end

function myautocov(M::AbstractMatrix, lags)
    D, T = size(M)
    return map(lags) do h
        Γ = zeros(eltype(M), D, D)
        @views for t in 1:(T - h)
            Γₜ = M[:, t + h] * M[:, t]'
            Γ .+= Γₜ
        end
        return Γ ./ (T - h)
    end
end

function fastautocov(M::AbstractMatrix, lags)
    result = crosscov(transpose(M), transpose(M), lags; demean=false)
    return [transpose(result[h, :, :]) for h in axes(result, 1)]
end

function empirical_covariances(
    proj::AbstractMatrix{Bool},
    Y::AbstractMatrix{<:Real},
    model::POVARModel,
    lags::AbstractVector{<:Integer},
)
    (; ω) = model
    D, T = size(Y)
    X̂ = proj .* Y
    all_Γs = fastautocov(X̂, lags)
    for (Γ, h) in zip(all_Γs, lags)
        Γ ./= scaling(model, h)
        if h == 0
            Γ .-= ω^2 * I(D)
        end
    end
    return all_Γs
end
