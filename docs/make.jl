using Documenter, MHDSim

makedocs(
    sitename = "MHDSim.jl",
    modules = [MHDSim],
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(
    repo = "https://github.com/aa4cc/MHDSim.jl",
    devbranch = "main",
)