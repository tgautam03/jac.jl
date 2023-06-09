using Documenter
using jac

makedocs(
    sitename = "jac",
    format = Documenter.HTML(),
    modules = [jac],
    pages = [
        "Home" => "index.md",
        "Development" => Any["dev_blog/basic_scalar_example.md",
        ],
        "Manual" => Any["user_manual/Tensor.md",
        ],
    ],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/tgautam03/jac.jl.git"
)
