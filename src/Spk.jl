module Spk

if !(@__DIR__() in LOAD_PATH)
    push!(LOAD_PATH, @__DIR__)
end

include("./spk-src/PSTH.jl")
include("./spk-src/Transform.jl")
include("./spk-src/Xcorr.jl")
include("./spk-src/Efficacy.jl")
include("./spk-src/Plot.jl")

end # end module
