module pagerank

include("./simplerank.jl")
import .simplerank: matrix_A, power_iteration
using Graphs
using LinearAlgebra

function graph_pagerank(graph, d, e, ϵ)
    @assert 0.0 < d <= 1.0
    Â = matrix_Â(graph, e)
    v = pagerank_power_iter_with_mass_re(Â, d, e, ϵ)
    v[v.<ϵ] .= 0.0
    return any(v .< 0) ? -v : v
end
graph_pagerank(graph, d) = graph_pagerank(graph, d, normalize(ones(nv(graph)), 1), 10E-15)

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
    return power_iteration(B)
end

function matrix_Â(graph, e)
    A = matrix_A(graph)
    n = nv(graph)
    c = zeros(n)
    c[outdegree(graph).==0] .= 1.0
    return A + e * c'
end
matrix_Â(graph) = matrix_Â(graph, normalize(ones(nv(graph)), 1))

end

