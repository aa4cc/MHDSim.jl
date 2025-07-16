using MHDSim

# Basic parameters
n_el = 4 # Number of electrodes
n_mag = 4 # Number of electromagnets
σ = 5 # Conductivity (S/m)
Δt = 0.25 # Timestep (s)

# Function to resolve the path to the electric and magnetic fields data.
function resolve_fields_path(name::String; base_dir="~/.julia/MHDSimFields", url_base::String)
    base_dir = expanduser(base_dir)
    path = joinpath(base_dir, name)
    if !isdir(path)
        println("Downloading field data '$name'...")
        mkpath(base_dir)
        archive = joinpath(base_dir, "$name.tar.gz")
        url = joinpath(url_base, "$name.tar.gz")
        Downloads.download(url, archive)
        run(`tar -xzf $archive -C $base_dir`)
    end
    return path
end

# Download if needed and load into simulation
path_to_fields = resolve_fields_path(
    "fields_8mm_4el_4mag";
    url_base = "https://github.com/aa4cc/MHDSim.jl/releases/download/v1.0"
)


# Initialize the simulation data

data = SimData(path_to_fields, n_el, n_mag, σ, Δt) # Takes a lot of time 

state = SimState(data) # Initial state of the simulation

## Points for evaluating the solution
xs = collect(LinRange(-50e-3, 50e-3, 32)) # x coordinates of the points where the fields will be evaluated
ys = xs # y coordinates of the points where the fields will be evaluated
zs = [8e-3] # z coordinates of the points where the fields will be evaluated

peh = MHDSim.PointEvalHandler(data, xs, ys, zs)


## Running the simulation for ten steps
cache = nothing
for i in 1:10

    ϕ = zeros(n_el) # Electrode commands
    ψ = zeros(n_mag) # Coil commands

    state, cache = solve_step(data, state, ϕ , ψ, cache)

    vx, vy, vz =  measure_velocity(state, data, peh) # Measure velocity at the points

    p = measure_pressure(state, data, peh) # Measure pressure at the points

    # Do something with the velocity
    # Do something with the pressure

end
