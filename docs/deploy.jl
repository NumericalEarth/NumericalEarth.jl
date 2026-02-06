using Documenter

@info "Cleaning up temporary .jld2 and .nc output created by doctests or literated examples..."

for (root, _, filenames) in walkdir(@__DIR__)
    for file in filenames
        if any(occursin(file), (r"\.jld2$", r"\.nc$"))
            rm(joinpath(root, file))
        end
    end
end

deploydocs(repo = "github.com/NumericalEarth/NumericalEarth.jl",
           deploy_repo = "github.com/NumericalEarth/NumericalEarthDocumentation.git",
           versions = ["stable" => "v^", "dev" => "dev", "v#.#.#" => "v#.#.#"],
           forcepush = true,
           devbranch = "main",
           push_preview = all(!isempty, (get(ENV, "GITHUB_TOKEN", ""), get(ENV, "DOCUMENTER_KEY", ""))))
