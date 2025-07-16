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
    repo = "github.com/aa4cc/MHDSim.jl.git",
    target = "build",
    branch = "gh-pages",
    devbranch = "main"
)