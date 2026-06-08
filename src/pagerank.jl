module pagerank

include("./simplerank.jl")
using .simplerank
using Graphs
using LinearAlgebra
using SparseArrays

function graph_pagerank(graph, d, e, ϵ=10E-15)
    @assert 0.0 < d <= 1.0
    Â = matrix_Â(graph, e)
    v = pagerank_power_iter_with_mass_re(Â, d, e, ϵ)
    v[v.<ϵ] .= 0.0
    return any(v .< 0) ? -v : v
end

function graph_pagerank(graph, d, e, ϵ, ::Val{:delta_data})
    @assert 0.0 < d <= 1.0
    Â = matrix_Â(graph, e)
    v, delta_v = pagerank_power_iter_with_mass_re(Â, d, e, ϵ, Val(:delta_data))
    v[v.<ϵ] .= 0.0
    return any(v .< 0) ? -v : v, delta_v
end

graph_pagerank(graph, d) = graph_pagerank(graph, d, normalize(ones(nv(graph)), 1), 10E-15)
graph_pagerank(graph, d, ::Val{:delta_data}) = graph_pagerank(graph, d, normalize(ones(nv(graph)), 1), 10E-15, Val(:delta_data))

function pagerank_power_iter_with_mass_re(A, d, e, ϵ)
    rᵢ = e
    rᵢ₊₁ = rᵢ
    δ = Inf
    while ϵ < δ
        rᵢ = rᵢ₊₁
        rᵢ₊₁ = d * A * rᵢ
        ρ = norm(rᵢ, 1) - norm(rᵢ₊₁, 1)
        rᵢ₊₁ += ρ * e
        δ = norm(rᵢ₊₁ - rᵢ, 1)
    end
    return rᵢ
end

function pagerank_power_iter_with_mass_re(A, d, e, ϵ, ::Val{:delta_data})
    rᵢ = e
    rᵢ₊₁ = rᵢ
    δ = Inf
    delta_v = []
    while ϵ < δ
        rᵢ = rᵢ₊₁
        rᵢ₊₁ = d * A * rᵢ
        ρ = norm(rᵢ, 1) - norm(rᵢ₊₁, 1)
        rᵢ₊₁ += ρ * e
        δ = norm(rᵢ₊₁ - rᵢ, 1)
        push!(delta_v, δ)
    end
    return rᵢ, delta_v
end

function graph_pagerank_basic(graph, d, e, ϵ)
    @assert 0.0 < d <= 1.0
    Â = matrix_Â(graph, e)
    v = pagerank_power_iter_basic(Â, d, e, ϵ)
    v[v.<ϵ] .= 0.0
    return any(v .< 0) ? -v : v
end
graph_pagerank_basic(graph, d) = graph_pagerank_basic(graph, d, normalize(ones(nv(graph)), 1), 10E-15)

function pagerank_power_iter_basic(A, d, e, ϵ)
    B = d * A + (1 - d) * e * ones(size(A)[1])'
    return simplerank.power_iteration(B)
end

function matrix_Â(graph, e)
    A = simplerank.matrix_A(graph)
    n = nv(graph)
    dangling = findall(x -> x == 0, outdegree(graph))

    m = length(dangling)
    I = Vector{Int}(undef, n * m)
    J = Vector{Int}(undef, n * m)
    V = Vector{Float64}(undef, n * m)

    idx = 1
    for col in dangling
        for row in 1:n
            I[idx] = row
            J[idx] = col
            V[idx] = e[row]
            idx += 1
        end
    end

    return A + sparse(I, J, V, n, n)
end
matrix_Â(graph) = matrix_Â(graph, normalize(ones(nv(graph)), 1))

end

