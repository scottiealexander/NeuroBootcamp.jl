
export relayed, isi_efficacy, efficacy

const TS = Vector{Float64}

# ============================================================================ #
function relayed(pre::TS, post::TS, kbin::AbstractArray{Int,1},
    bin_size::AbstractFloat, nbase::Integer, thr::AbstractFloat)

    # NOTE: we are looking at the spiking of pre relative to post, so that
    # the indicies in <grp> refer to the the pre spike train
    xcm, grp = psth(pre, post, kbin, bin_size, Binwise)
    xc = vec(sum(xcm, dims=2))

    kmax = findmax(xc)[2]
    baseline = xc[[1:nbase; end-(nbase-1):end]]
    threshold = mean(baseline) + (std(baseline) * thr)

    kstart = kmax - (findfirst(xc[kmax:-1:1] .< threshold) - 1)
    kstart = 1 <= kstart < kmax ? kstart : kmax

    kend = kmax + (findfirst(xc[kmax:end] .< threshold) - 1)
    kend = length(xc) >= kend > kmax ? kend : kmax

    return sort(unique(cat(grp[kstart:kend]..., dims=1)))
end
# ============================================================================ #
"""
`isi_efficacy(pre, post, deadtime=0.02, bin_size=0.001, isimax=0.03, max_delay=0.015)`
### Inputs:
* pre - timestamps of presynaptic spikes (in seconds)
* post - timestamps of postsynaptic spikes (in seconds)
* bin_size - psth/xcorr bin size (in seconds)
* isi_bin_size - bin size for isi discretization
* isimax - (0.03) max ISI length to include in the analysis (in seconds)
* baseline - (0.01) baseline window size (in seconds) used to construct threshold for identifying peak bins
* deadtime - (0.02) minimum deadtime preceeding first spike
* max_delay - (+/-0.015) the max synaptic delay to use to constrain search for xcorr peak (in seconds)

### Output:
* eff - a vector of % efficacy values
* lab - a vector of bin labels for <eff>

#### Notes:
* output <eff> will have NaN entries where the ISI distribution == 0
"""
function isi_efficacy(pre::TS, post::TS, bin_size::Float64,
    isi_bin_size::Float64, isimax::Float64=0.03, baseline::Float64=0.01,
    deadtime::Float64=0.02, max_delay::Float64=0.015)

    bin = round(Int64, (max_delay + baseline) / bin_size)
    kbin = -bin:+bin

    nbase = round(Int64, baseline / bin_size)

    # find all relayed spikes: by performing this operation on the full spike
    # train we minimize the liklihood of missing the true peak
    krel = relayed(pre, post, kbin, bin_size, nbase, 3.0)

    # preceeding isi (i.e. duration between each spike and the spike before it)
    # NOTE: the last spike cannot be the first spike of a pair (hence [1:end-1])
    pre_isi = [0.0; diff(pre[1:end-1])]

    # all pre spikes with a pre_isi >= deadtime (20ms in Usrey et al. 1998)
    first_spk = findall(pre_isi .>= deadtime)

    second_spk = first_spk .+ 1

    # relayed second spikes are simply the intersection of all second spikes and
    # all relayed spikes (indicies)
    krel_second = findall(in(krel), second_spk)

    # post isi (i.e. isi between first (<pre_isi> spikes) and second spikes)
    post_isi = pre[second_spk] - pre[first_spk]

    edges = 0.0:isi_bin_size:isimax+isi_bin_size

    # isi distribution of relayed second spikes divided by isi distribution of
    # all second spikes
    d = hist(post_isi[krel_second], edges)[2] ./ hist(post_isi, edges)[2]

    return d, edges[1:end-1]
end
# ============================================================================ #
"""
`efficacy(pre, post, evt, dur; bin_size=0.001, win=1,max_delay=0.015)`

Trialwise efficacy calculation

### Inputs:
* pre - vector of timestamps from presynaptic cell
* post - vector of timestamps from postsynaptic cell
* evt - vector of event timestamps
* dur - trial duration in seconds
* bin_size - (0.001) bin size in seconds
* baseline - (0.01) baseline window size (in seconds)
* max_delay - (0.015) the max synaptic delay in seconds

### Outputs:
* ef - a vector of efficacy values, one for each trial
"""
function efficacy(pre::TS, post::TS, evt::AbstractArray{<:Real}, dur::AbstractFloat,
    bin_size::AbstractFloat, baseline=0.01, max_delay=0.015)

    bin = round(Int64, (max_delay + baseline) ./ bin_size)
    kbin = -bin:+bin

    nbase = round(Int64, baseline / bin_size)

    # get indicies of all relayed spikes
    krel = relayed(pre, post, kbin, bin_size, nbase, 3.0)

    kbin = 0:round(Int64, dur / bin_size)

    pre_grp = psth(pre, evt, kbin, bin_size, Trialwise)[2]

    ef = zeros(length(pre_grp))
    @inbounds for k = 1:length(pre_grp)
        # ef[k] = length(findin(pre_grp[k], krel)) / length(pre_grp[k])
        ef[k] = length(findall(in(krel), pre_grp[k])) / length(pre_grp[k])
    end

    return ef
end
# ============================================================================ #
