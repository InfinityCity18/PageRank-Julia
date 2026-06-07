module plotting

export plot_pagerank

include("./pagerank.jl")
using Karnak, Plots
using .pagerank

function plot_pagerank(graph, d, method, type)
    function color_lerp(x, min_pr, max_pr)
        y₀ = 90.0 #green in hsv
        x₀ = min_pr
        y₁ = 0.0 #red in hsv
        x₁ = max_pr
        return y₀ + (x - x₀) * (y₁ - y₀) / (x₁ - x₀)
    end
    pagerank_v = round.(sigdigits=3, method(graph, d))
    @drawsvg begin
        background("grey10")
        sethue("white")
        drawgraph(graph,
            layout=type,
            vertexlabels=pagerank_v,
            vertexshapesizes=15,
            vertexfillcolors=
            [HSV(color_lerp(x, minimum(pagerank_v), maximum(pagerank_v)), 1.0, 1.0)
             for x in pagerank_v]
        )
    end 800 800
end

end