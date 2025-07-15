module MHDSim

using Ferrite, FerriteGmsh
using SparseArrays, BlockArrays
using UnPack
using IncompleteLU, Krylov
using NearestNeighbors
using Base.Threads
using JLD
using LinearAlgebra


export SimData, SimState, solve_step, measure_velocity, measure_pressure, save_force_vectors, PointEvalHandler


const ρ = 998.2 # Kg⋅m³
const ν = 1.0016e-6 # m²/s
const dim = 3


struct ScratchValues{T,CV<:CellValues,dim,Ti}
    Kₑ::Matrix{T}
    cellvalues::CV
    global_dofs::Vector{Int}
    coordinates::Vector{Vec{dim,T}}
    assembler::Ferrite.AssemblerSparsityPattern{T,Ti}
end

struct ScratchValuesVec{T,CV<:CellValues,dim,Ti}
    Kₑ::Matrix{T}
    fₑ::Vector{T}
    cellvalues::CV
    global_dofs::Vector{Int}
    coordinates::Vector{Vec{dim,T}}
    assembler::Ferrite.AssemblerSparsityPattern{T,Ti}
end



struct SimData{T<:Real,CT<:Ferrite.AbstractCell,CV_v<:CellValues,CV_p<:CellValues,FV_p<:FaceValues,Ti<:Int,dim}
    grid::Grid{dim,CT,T}
    colors::Array{Array{Int64,1},1}
    cellvalues_v::Vector{CV_v}
    cellvalues_p::Vector{CV_p}
    facevalues_p::FV_p
    dh_v::DofHandler{dim,Grid{dim,CT,T}}
    dh_p::DofHandler{dim,Grid{dim,CT,T}}
    ch_v::ConstraintHandler{DofHandler{dim,Grid{dim,CT,T}},T}
    ch_p::ConstraintHandler{DofHandler{dim,Grid{dim,CT,T}},T}
    M::SparseMatrixCSC{T,Ti}
    K::SparseMatrixCSC{T,Ti}
    G::SparseMatrixCSC{T,Ti}
    R::SparseMatrixCSC{T,Ti}
    A₀::SparseMatrixCSC{T,Ti}
    R_f::SparseArrays.UMFPACK.UmfpackLU{T,Ti}
    M_f::SparseArrays.CHOLMOD.Factor{T}
    rhsdata_M::Ferrite.RHSData{T}
    rhsdata_R::Ferrite.RHSData{T}
    f::Array{T,3}
    nϕ::Int
    nψ::Int
    Δt::T
end


struct SimState{T}
    uₙ::Vector{T}
    uₙ₋₁::Vector{T}
    pₙ::Vector{T}
    pₙ₋₁::Vector{T}
    pₙ₋₂::Vector{T}
end



function setup_cellvalues_and_dhs(grid::Grid{dim,Hexahedron,T}) where {dim,T}

    ip_v = Lagrange{dim,RefCube,2}()
    ip_geom = Lagrange{dim,RefCube,1}()
    qr = QuadratureRule{dim,RefCube}(4)

    qr_face = QuadratureRule{dim - 1,RefCube}(4)

    cellvalues_v = [CellVectorValues(qr, ip_v, ip_geom) for i in 1:Threads.nthreads()]

    ip_p = Lagrange{dim,RefCube,1}()
    cellvalues_p = [CellScalarValues(qr, ip_p, ip_geom) for i in 1:Threads.nthreads()]

    facevalues_p = FaceScalarValues(qr_face, ip_p, ip_geom)

    dh_v = DofHandler(grid)
    Ferrite.add!(dh_v, :v, dim, ip_v)

    close!(dh_v)

    dh_p = DofHandler(grid)
    Ferrite.add!(dh_p, :p, 1, ip_p)
    close!(dh_p)

    return cellvalues_v, cellvalues_p, facevalues_p, dh_v, dh_p
end


function setup_cellvalues_and_dhs(grid::Grid{dim,Tetrahedron,T}) where {dim,T}

    ip_v = Lagrange{dim,RefTetrahedron,2}()
    ip_geom = Lagrange{dim,RefTetrahedron,1}()
    qr = QuadratureRule{dim,RefTetrahedron}(4)

    qr_face = QuadratureRule{dim - 1,RefTetrahedron}(4)

    cellvalues_v = [CellVectorValues(qr, ip_v, ip_geom) for i in 1:Threads.nthreads()]



    ip_p = Lagrange{dim,RefTetrahedron,1}()
    cellvalues_p = [CellScalarValues(qr, ip_p, ip_geom) for i in 1:Threads.nthreads()]

    facevalues_p = FaceScalarValues(qr_face, ip_p, ip_geom)

    dh_v = DofHandler(grid)
    add!(dh_v, :v, dim, ip_v)

    close!(dh_v)

    dh_p = DofHandler(grid)
    add!(dh_p, :p, 1, ip_p)
    close!(dh_p)

    return cellvalues_v, cellvalues_p, facevalues_p, dh_v, dh_p

end


function setup_constraints(cellvalues_v, cellvalues_p, dh_v::DofHandler{dim}, dh_p::DofHandler{dim}, grid) where {dim}

    ## Velocity
    ch_v = ConstraintHandler(dh_v)
    nosplip_face_names = ["bottom", "sides"]
    ∂Ω_noslip = union(getfaceset.((grid,), nosplip_face_names)...)
    noslip_bc = Dirichlet(:v, ∂Ω_noslip, (x, t) -> [0.0, 0.0, 0.0], [1, 2, 3])
    add!(ch_v, noslip_bc)

    ∂Ω_top = getfaceset(grid, "top")

    top_bc = Dirichlet(:v, ∂Ω_top, (x, t) -> [0.0], [3])
    add!(ch_v, top_bc)

    close!(ch_v)
    Ferrite.update!(ch_v, 0.0)

    ## Pressure
    ch_p = ConstraintHandler(dh_p)

    addvertexset!(grid, "corner", (x) -> x[1] ≈ cos(0.0) * 71.5e-3 && x[2] ≈ sin(0.0) * 71.5e-3 && x[3] ≈ 0.0)


    ∂Ω_corner = getvertexset(grid, "corner")
    corner_bc = Dirichlet(:p, ∂Ω_corner, (x, t) -> [0.0], [1])
    add!(ch_p, corner_bc)

    close!(ch_p)
    Ferrite.update!(ch_p, 0.0)

    return ch_v, ch_p
end


function setup_mean_value_constraint(dh::DofHandler{dim}, cellvalues) where {dim}

    assembler = start_assemble()

    n_basefuncs = getnbasefunctions(cellvalues)

    Cₑ = zeros(1, n_basefuncs)

    for cell in CellIterator(dh)

        Ferrite.reinit!(cellvalues, cell)

        Cₑ .= 0

        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)

            for i = 1:n_basefuncs
                ψᵢ = shape_value(cellvalues, q_point, i)
                Cₑ[1, i] += ψᵢ * dΩ
            end
        end

        assemble!(assembler, [1], celldofs(cell), Cₑ)

    end

    C = end_assemble(assembler)

    # Create an AffineConstraint from the C-matrix
    _, J, V = findnz(C)
    _, constrained_dof_idx = findmax(abs2, V)
    constrained_dof = J[constrained_dof_idx]
    V ./= V[constrained_dof_idx]
    mean_value_constraint = AffineConstraint(
        constrained_dof,
        Pair{Int,Float64}[J[i] => -V[i] for i in 1:length(J) if J[i] != constrained_dof],
        0.0,
    )
    return mean_value_constraint
end


function create_scratchvalues(K, cellvalues, dh::DofHandler{dim}) where {dim}
    nthreads = Threads.nthreads()
    assemblers = [start_assemble(K) for i in 1:nthreads]

    n_basefuncs = getnbasefunctions(cellvalues[1])
    global_dofs = [zeros(Int, ndofs_per_cell(dh)) for i in 1:nthreads]

    Kₑs = [zeros(n_basefuncs, n_basefuncs) for i in 1:nthreads]

    coordinates = [[zero(Vec{dim}) for i in 1:length(dh.grid.cells[1].nodes)] for i in 1:nthreads]

    return [ScratchValues(Kₑs[i], cellvalues[i], global_dofs[i], coordinates[i], assemblers[i]) for i in 1:nthreads]
end


function create_scratchvalues(K, f, cellvalues, dh::DofHandler{dim}) where {dim}
    nthreads = Threads.nthreads()
    assemblers = [start_assemble(K, f) for i in 1:nthreads]

    n_basefuncs = getnbasefunctions(cellvalues[1])
    global_dofs = [zeros(Int, ndofs_per_cell(dh)) for i in 1:nthreads]

    Kₑs = [zeros(n_basefuncs, n_basefuncs) for i in 1:nthreads]
    fₑs = [zeros(n_basefuncs) for i in 1:nthreads]

    coordinates = [[zero(Vec{dim}) for i in 1:length(dh.grid.cells[1].nodes)] for i in 1:nthreads]

    return [ScratchValuesVec(Kₑs[i], fₑs[i], cellvalues[i], global_dofs[i], coordinates[i], assemblers[i]) for i in 1:nthreads]
end


function assemble_gradient_matrix(cellvalues_v::CellVectorValues{dim}, cellvalues_p::CellScalarValues{dim}, dh_v::DofHandler, dh_p::DofHandler)
    # Again, some buffers and helpers
    n_basefuncs_v = getnbasefunctions(cellvalues_v)
    n_basefuncs_p = getnbasefunctions(cellvalues_p)
    Gₑ = zeros(n_basefuncs_v, n_basefuncs_p)

    # Assembly loop
    gradient_assembler = start_assemble()
    for (cell_v, cell_p) in zip(CellIterator(dh_v), CellIterator(dh_p))
        # Don't forget to initialize everything
        fill!(Gₑ, 0)

        Ferrite.reinit!(cellvalues_v, cell_v)
        Ferrite.reinit!(cellvalues_p, cell_p)

        # Qudrature is the same for the pressure as for the velocity
        for q_point in 1:getnquadpoints(cellvalues_v)
            dΩ = getdetJdV(cellvalues_v, q_point)

            for j in 1:n_basefuncs_p
                ψ = shape_value(cellvalues_p, q_point, j)
                for i in 1:n_basefuncs_v
                    divφ = shape_divergence(cellvalues_v, q_point, i)
                    Gₑ[i, j] += (divφ * ψ) * dΩ
                end
            end
        end

        # Assemble `Gₑ` into the matrix `G`.
        assemble!(gradient_assembler, celldofs(cell_v), celldofs(cell_p), Gₑ)
    end

    G = end_assemble(gradient_assembler)

    return G
end

function assemble_product_matrix(cellvalues_v::CellVectorValues{dim}, cellvalues_p::CellScalarValues{dim}, dh_v::DofHandler, dh_p::DofHandler)
    # Again, some buffers and helpers
    n_basefuncs_v = getnbasefunctions(cellvalues_v)
    n_basefuncs_p = getnbasefunctions(cellvalues_p)
    Pₑ = zeros(n_basefuncs_p, n_basefuncs_v)

    # Assembly loop
    product_assembler = start_assemble()
    for (cell_v, cell_p) in zip(CellIterator(dh_v), CellIterator(dh_p))
        # Don't forget to initialize everything
        fill!(Pₑ, 0)

        Ferrite.reinit!(cellvalues_v, cell_v)
        Ferrite.reinit!(cellvalues_p, cell_p)

        # Qudrature is the same for the pressure as for the velocity
        for q_point in 1:getnquadpoints(cellvalues_v)
            dΩ = getdetJdV(cellvalues_v, q_point)

            for i in 1:n_basefuncs_v
                ∇ψᵢ = shape_gradient(cellvalues_p, q_point, i)
                for j in 1:n_basefuncs_v
                    φⱼ = shape_value(cellvalues_v, q_point, i)
                    Pₑ[i, j] += ∇ψᵢ ⋅ φⱼ * dΩ
                end
            end
        end

        # Assemble `Gₑ` into the matrix `G`.
        assemble!(product_assembler, celldofs(cell_p), celldofs(cell_v), Pₑ)
    end

    P = end_assemble(product_assembler)

    return P
end

function assemble_mass_matrix_cell!(scratch::ScratchValues, cell::Int, K::SparseMatrixCSC, grid::Grid, dh::DofHandler)

    Mₑ, cellvalues, global_dofs, coordinates, assembler =
        scratch.Kₑ, scratch.cellvalues, scratch.global_dofs, scratch.coordinates, scratch.assembler

    fill!(Mₑ, 0.0)

    n_basefuncs = getnbasefunctions(cellvalues)

    # Fill up the coordinates
    nodeids = grid.cells[cell].nodes
    for j in eachindex(coordinates)
        coordinates[j] = grid.nodes[nodeids[j]].x
    end

    reinit!(cellvalues, coordinates)

    for q_point in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)
        # Remember that we assemble a vector mass term, hence the dot product.
        for i in 1:n_basefuncs
            φᵢ = shape_value(cellvalues, q_point, i)
            for j in 1:n_basefuncs
                φⱼ = shape_value(cellvalues, q_point, j)
                Mₑ[i, j] += φᵢ ⋅ φⱼ * dΩ
            end
        end
    end
    celldofs!(global_dofs, dh, cell)
    assemble!(assembler, global_dofs, Mₑ)
end

function assemble_mass_matrix!(M::SparseMatrixCSC, colors, cellvalues, grid::Grid, dh::DofHandler{dim}) where {dim}
    scratches = create_scratchvalues(M, cellvalues, dh)
    for color in colors
        Threads.@threads :static for i in eachindex(color)
            assemble_mass_matrix_cell!(scratches[Threads.threadid()], color[i], M, grid, dh)
        end
    end
end

function assemble_stiffness_matrix_cell!(scratch::ScratchValues, cell::Int, ν, K::SparseMatrixCSC, grid::Grid, dh::DofHandler)

    Kₑ, cellvalues, global_dofs, coordinates, assembler =
        scratch.Kₑ, scratch.cellvalues, scratch.global_dofs, scratch.coordinates, scratch.assembler

    fill!(Kₑ, 0.0)

    n_basefuncs = getnbasefunctions(cellvalues)

    # Fill up the coordinates
    nodeids = grid.cells[cell].nodes
    for j in eachindex(coordinates)
        coordinates[j] = grid.nodes[nodeids[j]].x
    end

    reinit!(cellvalues, coordinates)

    for q_point in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:n_basefuncs
            ∇φᵢ = shape_gradient(cellvalues, q_point, i)
            for j in 1:n_basefuncs
                ∇φⱼ = shape_gradient(cellvalues, q_point, j)
                Kₑ[i, j] -= ν * ∇φᵢ ⊡ ∇φⱼ * dΩ
            end
        end
    end
    celldofs!(global_dofs, dh, cell)
    assemble!(assembler, global_dofs, Kₑ)
end

function assemble_stiffness_matrix!(K::SparseMatrixCSC, ν, colors, cellvalues, grid::Grid, dh::DofHandler{dim}) where {dim}
    scratches = create_scratchvalues(K, cellvalues, dh)
    for color in colors
        Threads.@threads :static for i in eachindex(color)
            assemble_stiffness_matrix_cell!(scratches[Threads.threadid()], color[i], ν, K, grid, dh)
        end
    end
end

function assemble_pressure_laplacian_matrix_cell!(scratch::ScratchValues, cell::Int, K::SparseMatrixCSC, grid::Grid, dh::DofHandler)

    Rₑ, cellvalues, global_dofs, coordinates, assembler =
        scratch.Kₑ, scratch.cellvalues, scratch.global_dofs, scratch.coordinates, scratch.assembler

    fill!(Rₑ, 0.0)

    n_basefuncs = getnbasefunctions(cellvalues)

    # Fill up the coordinates
    nodeids = grid.cells[cell].nodes
    for j in eachindex(coordinates)
        coordinates[j] = grid.nodes[nodeids[j]].x
    end

    reinit!(cellvalues, coordinates)

    for q_point in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:n_basefuncs
            ∇ψᵢ = shape_gradient(cellvalues, q_point, i)
            for j in 1:n_basefuncs
                ∇ψⱼ = shape_gradient(cellvalues, q_point, j)
                Rₑ[i, j] += ∇ψᵢ ⋅ ∇ψⱼ * dΩ
            end
        end
    end
    celldofs!(global_dofs, dh, cell)
    assemble!(assembler, global_dofs, Rₑ)
end

function assemble_pressure_laplacian_matrix!(R::SparseMatrixCSC, colors, cellvalues, grid::Grid, dh::DofHandler{dim}) where {dim}
    scratches = create_scratchvalues(R, cellvalues, dh)
    for color in colors
        Threads.@threads :static for i in eachindex(color)
            assemble_pressure_laplacian_matrix_cell!(scratches[Threads.threadid()], color[i], R, grid, dh)
        end
    end
end

function assemble_convective_matrix_cell!(scratch::ScratchValues, cell::Int, uₙ₋₂::Vector{Float64}, uₙ₋₁::Vector{Float64}, grid::Grid, dh::DofHandler)

    Cₑ, cellvalues, global_dofs, coordinates, assembler =
        scratch.Kₑ, scratch.cellvalues, scratch.global_dofs, scratch.coordinates, scratch.assembler

    fill!(Cₑ, 0.0)

    n_basefuncs = getnbasefunctions(cellvalues)

    # Fill up the coordinates
    nodeids = grid.cells[cell].nodes
    for j in eachindex(coordinates)
        coordinates[j] = grid.nodes[nodeids[j]].x
    end

    celldofs!(global_dofs, dh, cell)

    v_cellₙ₋₁ = uₙ₋₁[global_dofs]
    v_cellₙ₋₂ = uₙ₋₂[global_dofs]

    reinit!(cellvalues, coordinates)

    for q_point in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)

        vₙ₋₁ = function_value(cellvalues, q_point, v_cellₙ₋₁)
        vₙ₋₂ = function_value(cellvalues, q_point, v_cellₙ₋₂)


        v̄ = -2 * vₙ₋₁ + vₙ₋₂

        for j in 1:n_basefuncs
            ∇φⱼ = shape_gradient(cellvalues, q_point, j)
            φⱼ = shape_value(cellvalues, q_point, j)
            for i in 1:n_basefuncs
                φᵢ = shape_value(cellvalues, q_point, i)
                Cₑ[i, j] += (v̄ ⋅ ∇φⱼ') ⋅ φᵢ * dΩ
            end
        end
    end
    assemble!(assembler, global_dofs, Cₑ)
end

function assemble_convective_matrix!(C::SparseMatrixCSC, uₙ₋₂::Vector{Float64}, uₙ₋₁::Vector{Float64}, colors, cellvalues, grid::Grid, dh::DofHandler{dim}) where {dim}
    scratches = create_scratchvalues(C, cellvalues, dh)
    for color in colors
        Threads.@threads :static for i in eachindex(color)
            assemble_convective_matrix_cell!(scratches[Threads.threadid()], color[i], uₙ₋₂::Vector{Float64}, uₙ₋₁::Vector{Float64}, grid, dh)
        end
    end
end



function assemble_convective_term!(J::SparseMatrixCSC, f::Vector{T}, u::Vector{T}, colors, cellvalues, grid::Grid, dh::DofHandler{dim}) where {dim,T<:Real}
    scratches = create_scratchvalues(J, f, cellvalues, dh)
    for color in colors
        Threads.@threads :static for i in eachindex(color)
            assemble_convective_term_cell!(scratches[Threads.threadid()], color[i], u, grid, dh)
        end
    end
end


function assemble_convective_term_cell!(scratch::ScratchValuesVec, cell::Int, u::Vector{<:Real}, grid::Grid, dh::DofHandler)

    Jₑ, fₑ, cellvalues, global_dofs, coordinates, assembler =
        scratch.Kₑ, scratch.fₑ, scratch.cellvalues, scratch.global_dofs, scratch.coordinates, scratch.assembler

    fill!(Jₑ, 0.0)
    fill!(fₑ, 0.0)

    n_basefuncs = getnbasefunctions(cellvalues)

    # Fill up the coordinates
    nodeids = grid.cells[cell].nodes
    for j in eachindex(coordinates)
        coordinates[j] = grid.nodes[nodeids[j]].x
    end

    celldofs!(global_dofs, dh, cell)

    v_cell = u[global_dofs]

    reinit!(cellvalues, coordinates)

    for q_point in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)


        v = function_value(cellvalues, q_point, v_cell)
        ∇v = function_gradient(cellvalues, q_point, v_cell)

        for j in 1:n_basefuncs
            ∇φⱼ = shape_gradient(cellvalues, q_point, j)
            φⱼ = shape_value(cellvalues, q_point, j)

            for i in 1:n_basefuncs
                φᵢ = shape_value(cellvalues, q_point, i)
                Jₑ[i, j] -= (v ⋅ ∇φⱼ') ⋅ φᵢ * dΩ
                Jₑ[i, j] -= (φⱼ ⋅ ∇v') ⋅ φᵢ * dΩ
            end

            fₑ[j] -= (v ⋅ ∇v') ⋅ φⱼ * dΩ
        end
    end
    assemble!(assembler, global_dofs, Jₑ, fₑ)
end

function assemble_force_vectors(path_to_fields, nϕ::Int, nψ::Int, cellvalues_v::CellVectorValues{dim}, dh::DofHandler, σ::T) where {T<:Real}

    f = zeros(nϕ, nψ, ndofs(dh))

    for i = 1:nϕ
        Ex = load(joinpath(path_to_fields, "E$i.jld"), "Ex")
        Ey = load(joinpath(path_to_fields, "E$i.jld"), "Ey")
        Ez = load(joinpath(path_to_fields, "E$i.jld"), "Ez")
        coords_E = load(joinpath(path_to_fields, "E$i.jld"), "coords")

        tree_E = KDTree(coords_E, NearestNeighbors.Euclidean())


        for j = 1:nψ

            Bx = load(joinpath(path_to_fields, "B$j.jld"), "Bx")
            By = load(joinpath(path_to_fields, "B$j.jld"), "By")
            Bz = load(joinpath(path_to_fields, "B$j.jld"), "Bz")
            coords_B = load(joinpath(path_to_fields, "B$j.jld"), "coords")

            tree_B = KDTree(coords_B, NearestNeighbors.Euclidean())


            n_basefuncs = getnbasefunctions(cellvalues_v)
            for cell in CellIterator(dh)
                Ferrite.reinit!(cellvalues_v, cell)
                all_celldofs = celldofs(cell)
                v_celldofs = all_celldofs[dof_range(dh, :v)]
                for q_point in 1:getnquadpoints(cellvalues_v)
                    dΩ = getdetJdV(cellvalues_v, q_point)
                    x = spatial_coordinate(cellvalues_v, q_point, getcoordinates(cell))

                    idx_E, _ = nn(tree_E, x)
                    idx_B, _ = nn(tree_B, x)


                    Ex_val = Ex[idx_E]
                    Ey_val = Ey[idx_E]
                    Ez_val = Ez[idx_E]


                    Bx_val = Bx[idx_B]
                    By_val = By[idx_B]
                    Bz_val = Bz[idx_B]

                    # Ohm's law J = σE
                    Jx_val = σ * Ex_val
                    Jy_val = σ * Ey_val
                    Jz_val = σ * Ez_val

                    # Cross product J × B

                    Fx = Jy_val * Bz_val - Jz_val * By_val
                    Fy = Jz_val * Bx_val - Jx_val * Bz_val
                    Fz = Jx_val * By_val - Jy_val * Bx_val


                    F = Vec{3}((Fx, Fy, Fz))

                    for k in 1:n_basefuncs
                        φₖ = shape_value(cellvalues_v, q_point, k)

                        f[i, j, v_celldofs[k]] += 1 / ρ * F ⋅ φₖ * dΩ
                    end
                end
            end
        end
    end
    return f
end

function PointEvalHandler(data::SimData, xs::AbstractVector, ys::AbstractVector, zs::AbstractVector)

    points = Array{Vec{dim,Float64}}(undef, (length(xs), length(ys), length(zs)))

    for i = eachindex(xs)
        for j = eachindex(ys)
            for k = eachindex(zs)
                points[i, j, k] = Vec{3,Float64}((xs[i], ys[j], zs[k]))
            end
        end
    end

    points = vec(points)
    Ferrite.PointEvalHandler(data.grid, points)
end


function measure_velocity(state, data, peh)
    tmp = hcat(get_point_values(peh, data.dh_v, state.uₙ, :v)...)
    u, v, w = vec(tmp[1, :]), vec(tmp[2, :]), vec(tmp[3, :])
end

function measure_pressure(state, data, peh)
    p = get_point_values(peh, data.dh_p, state.pₙ, :p)
end


function SimData(path_to_fields, nϕ, nψ, σ, Δt; reuse_force_vectors=false, force_vectors_path="force_vectors.jld")

    grid = generate_grid(0.0, 8e-3, 71.5e-3, 50e-3, 0.8, 4, 15, 15)

    colors = create_coloring(grid)

    cellvalues_v, cellvalues_p, facevalues_p, dh_v, dh_p = setup_cellvalues_and_dhs(grid)

    @info "Number of velocity DOFs solved for: $(ndofs(dh_v))"
    @info "Number of pressure DOFs solved for: $(ndofs(dh_p))"

    ch_v, ch_p = setup_constraints(cellvalues_v, cellvalues_p, dh_v, dh_p, grid)

    M = create_sparsity_pattern(dh_v, ch_v)
    assemble_mass_matrix!(M, colors, cellvalues_v, grid, dh_v)

    M̃ = copy(M)
    rhsdata_M = get_rhs_data(ch_v, M̃)
    apply!(M̃, ch_v)
    M_f = cholesky(M̃)

    K = create_sparsity_pattern(dh_v, ch_v)
    assemble_stiffness_matrix!(K, ν, colors, cellvalues_v, grid, dh_v)

    A₀ = copy(3 / (2 * Δt) * M .- K)

    G = assemble_gradient_matrix(cellvalues_v[1], cellvalues_p[1], dh_v, dh_p)

    R = create_sparsity_pattern(dh_p, ch_p)
    assemble_pressure_laplacian_matrix!(R, colors, cellvalues_p, grid, dh_p)

    apply!(R, ch_p)
    rhsdata_R = get_rhs_data(ch_p, R)
    R_f = lu(R)

    if reuse_force_vectors
        f = load(force_vectors_path, "f")
        nϕ = load(force_vectors_path, "nphi")
        nψ = load(force_vectors_path, "npsi")
    else
        f = assemble_force_vectors(path_to_fields, nϕ, nψ, cellvalues_v[1], dh_v, σ)
        save(force_vectors_path, "f", f, "nphi", nϕ, "npsi", nψ)
    end

    data = SimData(grid, colors, cellvalues_v, cellvalues_p, facevalues_p, dh_v, dh_p, ch_v, ch_p, M, K, G, R, A₀, R_f, M_f, rhsdata_M, rhsdata_R, f, nϕ, nψ, Δt)

    @info "Simulation data has been set up"

    return data
end


function SimData(force_vectors_path, Δt;)

    grid = generate_grid(0.0, 8e-3, 71.5e-3, 50e-3, 0.8, 4, 15, 15)

    colors = create_coloring(grid)

    cellvalues_v, cellvalues_p, facevalues_p, dh_v, dh_p = setup_cellvalues_and_dhs(grid)

    @info "Number of velocity DOFs solved for: $(ndofs(dh_v))"
    @info "Number of pressure DOFs solved for: $(ndofs(dh_p))"

    ch_v, ch_p = setup_constraints(cellvalues_v, cellvalues_p, dh_v, dh_p, grid)


    M = create_sparsity_pattern(dh_v, ch_v)
    assemble_mass_matrix!(M, colors, cellvalues_v, grid, dh_v)

    M̃ = copy(M)
    rhsdata_M = get_rhs_data(ch_v, M̃)
    apply!(M̃, ch_v)
    M_f = cholesky(M̃)

    K = create_sparsity_pattern(dh_v, ch_v)
    assemble_stiffness_matrix!(K, ν, colors, cellvalues_v, grid, dh_v)

    A₀ = copy(3 / (2 * Δt) * M .- K)

    G = assemble_gradient_matrix(cellvalues_v[1], cellvalues_p[1], dh_v, dh_p)

    R = create_sparsity_pattern(dh_p, ch_p)
    assemble_pressure_laplacian_matrix!(R, colors, cellvalues_p, grid, dh_p)

    apply!(R, ch_p)
    rhsdata_R = get_rhs_data(ch_p, R)
    R_f = lu(R)

    f = load(force_vectors_path, "f")
    nϕ = load(force_vectors_path, "nphi")
    nψ = load(force_vectors_path, "npsi")

    data = SimData(grid, colors, cellvalues_v, cellvalues_p, facevalues_p, dh_v, dh_p, ch_v, ch_p, M, K, G, R, A₀, R_f, M_f, rhsdata_M, rhsdata_R, f, nϕ, nψ, Δt)

    @info "Simulation data has been set up"

    return data
end

function save_force_vectors(data::SimData, path)
    save(path, "f", data.f, "nphi", data.nϕ, "npsi", data.nψ)
end

function SimState(data::SimData)
    SimState(zeros(ndofs(data.dh_v)), zeros(ndofs(data.dh_v)), zeros(ndofs(data.dh_p)), zeros(ndofs(data.dh_p)), zeros(ndofs(data.dh_p)))
end


function solve_step(data::SimData, state::SimState, ϕ::Vector{T}, ψ::Vector{T}, cache=nothing) where {T<:Real}

    @unpack grid, colors, cellvalues_v, cellvalues_p, facevalues_p, dh_v, dh_p, ch_v, ch_p, M, K, G, R, A₀, R_f, M_f, rhsdata_M, rhsdata_R, f, nϕ, nψ, Δt = data

    uₙ₋₁, uₙ₋₂, pₙ₋₁, pₙ₋₂, pₙ₋₃ = state.uₙ, state.uₙ₋₁, state.pₙ, state.pₙ₋₁, state.pₙ₋₂


    if isnothing(cache)
        A = create_sparsity_pattern(dh_v, ch_v)
        C = create_sparsity_pattern(dh_v, ch_v)
        fₜ = zeros(ndofs(dh_v))
        b = zeros(ndofs(dh_v))
        q = zeros(ndofs(dh_p))
        g = zeros(ndofs(dh_v))
        uₙ = zeros(ndofs(dh_v))
        pₙ = zeros(ndofs(dh_p))
        δp = zeros(ndofs(dh_p))

    else
        A, C, fₜ, b, q, g, uₙ, pₙ, δp = cache
    end

    fₜ .= 0.0

    for j = 1:nψ
        for i = 1:nϕ
            @inbounds fₜ .+= ϕ[i] * ψ[j] * f[i, j, :]
        end
    end

    assemble_convective_matrix!(C, uₙ₋₂, uₙ₋₁, colors, cellvalues_v, grid, dh_v)

    A .= A₀
    A .-= C

    p̂ = 7 / 3 * pₙ₋₁ - 5 / 3 * pₙ₋₂ + 1 / 3 * pₙ₋₃

    b .= 2 / (Δt) * M * uₙ₋₁ - 1 / (2 * Δt) * M * uₙ₋₂ + G * p̂ + fₜ

    apply!(A, b, ch_v)

    P = ilu(A)

    (uₙ, stats) = Krylov.gmres(A, b, uₙ₋₁; M=P, ldiv=true, restart=true)

    apply!(uₙ, ch_v)


    q .= -3 / (2 * Δt) * G' * uₙ + R * pₙ₋₁

    apply_rhs!(rhsdata_R, q, ch_p)

    pₙ = R_f \ q - ν * G' * uₙ

    apply!(pₙ, ch_p)

    if !all(isfinite.(uₙ))
        throw(InvalidStateException("Nonfinite value(s) detected in the solution!", :nonfinite))
    end


    new_cache = A, C, fₜ, b, q, g, uₙ, pₙ, δp
    new_state = SimState(uₙ, uₙ₋₁, pₙ, pₙ₋₁, pₙ₋₂)

    new_state, new_cache
end




function generate_grid(h, height, r, len, alpha, n_z, n1, n2)

    gmsh.initialize()


    gmsh.option.set_number("General.Verbosity", 2)


    o = gmsh.model.geo.add_point(0.0, 0.0, 0.0, h)

    o_top = gmsh.model.geo.add_point(0.0, 0.0, height, h)


    angles = 0:pi/4:2*pi-pi/4


    bottom_circle_points = []
    top_circle_points = []


    for theta in angles
        x = r * cos(theta)
        y = r * sin(theta)

        p = gmsh.model.geo.add_point(x, y, 0.0, h)
        push!(bottom_circle_points, p)

        p = gmsh.model.geo.add_point(x, y, height, h)
        push!(top_circle_points, p)

    end

    bottom_circle_arcs = []
    top_circle_arcs = []

    n = length(angles)

    for i = 1:n

        idx = i
        next_idx = mod(i, n) + 1

        p1 = bottom_circle_points[idx]

        p2 = bottom_circle_points[next_idx]

        c = gmsh.model.geo.add_circle_arc(p1, o, p2)

        push!(bottom_circle_arcs, c)

        p1 = top_circle_points[idx]

        p2 = top_circle_points[next_idx]

        c = gmsh.model.geo.add_circle_arc(p1, o_top, p2)

        push!(top_circle_arcs, c)

    end


    bottom_inner_points = []
    top_inner_points = []


    for theta in angles

        if (theta / pi) % 0.5 ≈ 0.0

            x = alpha * sqrt(2) * len * cos(theta) / 2
            y = alpha * sqrt(2) * len * sin(theta) / 2

        else

            x = sqrt(2) * len * cos(theta) / 2
            y = sqrt(2) * len * sin(theta) / 2

        end

        p = gmsh.model.geo.add_point(x, y, 0.0, h)
        push!(bottom_inner_points, p)

        p = gmsh.model.geo.add_point(x, y, height, h)
        push!(top_inner_points, p)

    end

    bottom_inner_lines = []
    top_inner_lines = []

    n = length(angles)

    for i = 1:n

        idx = i
        next_idx = mod(i, n) + 1

        p1 = bottom_inner_points[idx]

        p2 = bottom_inner_points[next_idx]

        l = gmsh.model.geo.add_line(p1, p2)

        push!(bottom_inner_lines, l)

        p1 = top_inner_points[idx]

        p2 = top_inner_points[next_idx]

        l = gmsh.model.geo.add_line(p1, p2)

        push!(top_inner_lines, l)

    end

    bottom_to_top_line = gmsh.model.geo.add_line(o, o_top)

    bottom_connecting_lines = []
    top_connecting_lines = []


    for i = 1:n

        p1 = bottom_inner_points[i]
        p2 = bottom_circle_points[i]


        l = gmsh.model.geo.add_line(p1, p2)

        push!(bottom_connecting_lines, l)


        p1 = top_inner_points[i]

        p2 = top_circle_points[i]

        l = gmsh.model.geo.add_line(p1, p2)

        push!(top_connecting_lines, l)

    end


    center_bottom_connecting_lines = []
    center_top_connecting_lines = []

    for i = 1:2:n


        p1 = bottom_inner_points[i]

        l = gmsh.model.geo.add_line(o, p1)

        push!(center_bottom_connecting_lines, l)

        p1 = top_inner_points[i]

        l = gmsh.model.geo.add_line(o_top, p1)

        push!(center_top_connecting_lines, l)

    end



    outer_vertical_lines = []

    inner_vertical_lines = []

    for i = 1:n

        p1 = bottom_circle_points[i]

        p2 = top_circle_points[i]

        l = gmsh.model.geo.add_line(p1, p2)

        push!(outer_vertical_lines, l)

        p1 = bottom_inner_points[i]

        p2 = top_inner_points[i]

        l = gmsh.model.geo.add_line(p1, p2)

        push!(inner_vertical_lines, l)

    end


    outer_vertical_surfaces = []
    inner_vertical_surfaces = []
    connecting_vertical_surfaces = []


    for i = 1:n

        idx = i
        next_idx = mod(i, n) + 1

        cl = gmsh.model.geo.add_curve_loop([bottom_circle_arcs[idx], outer_vertical_lines[next_idx], -top_circle_arcs[idx], -outer_vertical_lines[idx]])
        s = gmsh.model.geo.add_surface_filling([cl])
        push!(outer_vertical_surfaces, s)

        cl = gmsh.model.geo.add_curve_loop([bottom_inner_lines[idx], inner_vertical_lines[next_idx], -top_inner_lines[idx], -inner_vertical_lines[idx]])
        s = gmsh.model.geo.add_plane_surface([cl])
        push!(inner_vertical_surfaces, s)


        cl = gmsh.model.geo.add_curve_loop([bottom_connecting_lines[idx], outer_vertical_lines[idx], -top_connecting_lines[idx], -inner_vertical_lines[idx]])

        s = gmsh.model.geo.add_plane_surface([cl])
        push!(connecting_vertical_surfaces, s)

    end


    bottom_outer_surfaces = []
    top_outer_surfaces = []

    for i = 1:n
        idx = i
        next_idx = mod(i, n) + 1

        cl = gmsh.model.geo.add_curve_loop([bottom_connecting_lines[idx], bottom_circle_arcs[idx], -bottom_connecting_lines[next_idx], -bottom_inner_lines[idx]])
        s = gmsh.model.geo.add_plane_surface([cl])
        push!(bottom_outer_surfaces, s)

        cl = gmsh.model.geo.add_curve_loop([top_connecting_lines[idx], top_circle_arcs[idx], -top_connecting_lines[next_idx], -top_inner_lines[idx]])
        s = gmsh.model.geo.add_plane_surface([cl])
        push!(top_outer_surfaces, s)


    end

    bottom_inner_surfaces = []
    top_inner_surfaces = []

    for i = 1:n÷2


        idx = i
        next_idx = mod(i, n ÷ 2) + 1

        cl = gmsh.model.geo.add_curve_loop([center_bottom_connecting_lines[idx], bottom_inner_lines[2*(idx-1)+1], bottom_inner_lines[2*(idx-1)+2], -center_bottom_connecting_lines[next_idx]])
        s = gmsh.model.geo.add_plane_surface([cl])
        push!(bottom_inner_surfaces, s)

        cl = gmsh.model.geo.add_curve_loop([center_top_connecting_lines[idx], top_inner_lines[2*(idx-1)+1], top_inner_lines[2*(idx-1)+2], -center_top_connecting_lines[next_idx]])
        s = gmsh.model.geo.add_plane_surface([cl])
        push!(top_inner_surfaces, s)

    end


    center_vertical_surfaces = []


    for i = 1:n÷2

        idx = i
        next_idx = mod(i, n ÷ 2) + 1

        cl = gmsh.model.geo.add_curve_loop([center_bottom_connecting_lines[idx], inner_vertical_lines[2*(idx-1)+1], -center_top_connecting_lines[idx], -bottom_to_top_line])
        s = gmsh.model.geo.add_plane_surface([cl])
        push!(center_vertical_surfaces, s)

    end


    outer_volumes = []


    for i = 1:n

        idx = i
        next_idx = mod(i, n) + 1

        sl = gmsh.model.geo.add_surface_loop([outer_vertical_surfaces[idx], bottom_outer_surfaces[idx], top_outer_surfaces[idx], connecting_vertical_surfaces[idx], connecting_vertical_surfaces[next_idx], inner_vertical_surfaces[idx]])

        v = gmsh.model.geo.add_volume([sl])
        push!(outer_volumes, v)



    end

    inner_volumes = []
    for i = 1:n÷2

        idx = i
        next_idx = mod(i, n ÷ 2) + 1

        sl = gmsh.model.geo.add_surface_loop([center_vertical_surfaces[idx], bottom_inner_surfaces[idx], top_inner_surfaces[idx], center_vertical_surfaces[next_idx], inner_vertical_surfaces[2*(idx-1)+1], inner_vertical_surfaces[2*(idx-1)+2]])

        v = gmsh.model.geo.add_volume([sl])
        push!(inner_volumes, v)

    end


    side_surfaces = outer_vertical_surfaces
    top_surfaces = vcat(top_outer_surfaces, top_inner_surfaces)
    bottom_surfaces = vcat(bottom_outer_surfaces, bottom_inner_surfaces)


    vertical_lines = vcat(outer_vertical_lines, inner_vertical_lines, bottom_to_top_line)

    transfinite_lines1 = vcat(bottom_inner_lines, top_inner_lines, center_bottom_connecting_lines, center_top_connecting_lines, bottom_circle_arcs, top_circle_arcs)
    transfinite_lines2 = vcat(bottom_connecting_lines, top_connecting_lines)



    transfinite_surfaces = vcat(outer_vertical_surfaces, inner_vertical_surfaces, bottom_outer_surfaces, top_outer_surfaces, center_vertical_surfaces, top_inner_surfaces, bottom_inner_surfaces, connecting_vertical_surfaces, center_vertical_surfaces)

    transfinite_volumes = vcat(outer_volumes, inner_volumes)




    gmsh.model.geo.synchronize()


    gmsh.model.add_physical_group(2, side_surfaces, -1, "sides")
    gmsh.model.add_physical_group(2, top_surfaces, -1, "top")
    gmsh.model.add_physical_group(2, bottom_surfaces, -1, "bottom")
    gmsh.model.add_physical_group(3, transfinite_volumes, -1, "Domain")


    for line in vertical_lines

        gmsh.model.mesh.set_transfinite_curve(line, n_z)

    end


    for line in transfinite_lines1

        gmsh.model.mesh.set_transfinite_curve(line, n1)

    end

    for line in transfinite_lines2

        gmsh.model.mesh.set_transfinite_curve(line, n2, "Bump", 0.2)

    end


    for surface in transfinite_surfaces

        gmsh.model.mesh.set_transfinite_surface(surface)

    end


    for volume in transfinite_volumes

        gmsh.model.mesh.set_transfinite_volume(volume)

    end



    for surface in transfinite_surfaces

        gmsh.model.mesh.set_recombine(2, surface)

    end

    gmsh.model.mesh.recombine()


    gmsh.model.mesh.generate(1)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.generate(3)


    dim = Int64(gmsh.model.getDimension())
    facedim = dim - 1

    # do stuff to describe your gmsh model

    # renumber the gmsh entities, such that the final used entities start by index 1
    # this step is crucial, because Ferrite.jl uses the implicit entity index based on an array index
    # and thus, need to start at 1
    gmsh.model.mesh.renumberNodes()
    gmsh.model.mesh.renumberElements()

    # transfer the gmsh information
    nodes = tonodes()
    elements, gmsh_elementidx = toelements(dim)
    cellsets = tocellsets(dim, gmsh_elementidx)

    # "Domain" is the name of a PhysicalGroup and saves all cells that define the computational domain
    domaincellset = cellsets["Domain"]
    elements = elements[collect(domaincellset)]

    boundarydict = toboundary(facedim)
    facesets = tofacesets(boundarydict, elements)
    gmsh.finalize()

    grid = Grid(elements, nodes, facesets=facesets, cellsets=cellsets)

    grid

end


end # module MHDSim
