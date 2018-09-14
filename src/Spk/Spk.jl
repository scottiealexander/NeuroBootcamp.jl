module Spk

using Statistics, FFTW, LinearAlgebra
using ..Histogram

const AbstractTS = AbstractVector{<:Real}

include("./PSTH.jl")
include("./Transform.jl")
include("./Xcorr.jl")
include("./Efficacy.jl")
#include("./Plot.jl")

end # end module
