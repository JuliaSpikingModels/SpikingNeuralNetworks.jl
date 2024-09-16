@snn_kw struct Receptor{T = Float32}
    E_rev::T = 0.0
    τr::T = -1.0f0
    τd::T = -1.0f0
    g0::T = 0.0f0
    gsyn::T = g0 > 0 ? g0 * norm_synapse(τr, τd) : 0.0f0
    α::T = α_synapse(τr, τd)
    τr⁻::T = 1 / τr > 0 ? 1 / τr : 0.0f0
    τd⁻::T = 1 / τd > 0 ? 1 / τd : 0.0f0
    nmda::T = 0.0f0
end

ReceptorVoltage = Receptor
SynapseArray = Vector{Receptor{Float32}}

@snn_kw struct NMDAVoltageDependency{T<:Float32}
    b::T = nmda_b
    k::T = nmda_k
    mg::T = Mg_mM
end
Mg_mM = 1.0f0
nmda_b = 3.36       #(no unit) parameters for voltage dependence of nmda channels
# nmda_k   = -0.062     #(1/V) source: http://dx.doi.org/10.1016/j.neucom.2011.04.018)
nmda_k = -0.077     #Eyal 2018

export Mg_mM, nmda_b, nmda_k


struct Synapse{T<:Receptor}
    AMPA::T
    NMDA::T
    GABAa::T
    GABAb::T
end

struct Glutamatergic{T<:Receptor}
    AMPA::T
    NMDA::T
end

struct GABAergic{T<:Receptor}
    GABAa::T
    GABAb::T
end

function Synapse(glu::Glutamatergic, gaba::GABAergic)
    return Synapse(glu.AMPA, glu.NMDA, gaba.GABAa, gaba.GABAb)
end

export Receptor,
    Synapse, ReceptorVoltage, GABAergic, Glutamatergic, SynapseArray, NMDAVoltageDependency

function norm_synapse(synapse::Receptor)
    norm_synapse(synapse.τr, synapse.τd)
end


function norm_synapse(τr, τd)
    p = [1, τr, τd]
    t_p = p[2] * p[3] / (p[3] - p[2]) * log(p[3] / p[2])
    return 1 / (-exp(-t_p / p[2]) + exp(-t_p / p[3]))
end

# α is the factor that has to be placed in-front of the differential equation as such the analytical integration corresponds to the double exponential function. Further details are discussed in the Julia notebook about synapses
function α_synapse(τr, τd)
    return (τd - τr) / (τd * τr)
end

function synapsearray(syn::Synapse, indices::Vector = [])::SynapseArray
    container = SynapseArray()
    names = isempty(indices) ? fieldnames(Synapse) : fieldnames(Synapse)[indices]
    for name in names
        receptor = getfield(syn, name)
        if receptor.gsyn > 0
            push!(container, receptor)
        end
    end
    return container
end


export norm_synapse

#############################################################
###########       Synapse parameters        #################
#############################################################


"""
Christof Koch. Biophysics of Computation: Information Processing in Single Neurons, by.Trends in Neurosciences, 22842(7):328–329, July 1999. ISSN 0166-2236, 1878-108X. doi: 10.1016/S0166-2236(99)01403-4.
"""
KochGlu =
    Glutamatergic(Receptor(E_rev = 0.00, τr = 0.2, τd = 25.0, g0 = 0.73), ReceptorVoltage())


"""
Guy Eyal, Matthijs B. Verhoog, Guilherme Testa-Silva, Yair Deitcher, Ruth Benavides-Piccione, Javier DeFelipe, Chris-832tiaan P. J. de Kock, Huibert D. Mansvelder, and Idan Segev. Human Cortical Pyramidal Neurons: From Spines to833Spikes via Models.Frontiers in Cellular Neuroscience, 12, 2018. ISSN 1662-5102. doi: 10.3389/fncel.2018.00181.
"""

EyalGluDend = Glutamatergic(
    Receptor(E_rev = 0.0, τr = 0.25, τd = 2.0, g0 = 0.73),
    ReceptorVoltage(E_rev = 0.0, τr = 8, τd = 35.0, g0 = 1.31, nmda = 1.0f0),
)

Mg_mM = 1.0f0
nmda_b = 3.36       #(no unit) parameters for voltage dependence of nmda channels
# nmda_k   = -0.062     #(1/V) source: http://dx.doi.org/10.1016/j.neucom.2011.04.018)
nmda_k = -0.077     #Eyal 2018
EyalNMDA = NMDAVoltageDependency(mg = Mg_mM, b = nmda_b, k = nmda_k)

EyalGluDend_nonmda = Glutamatergic(
    Receptor(E_rev = 0.0, τr = 0.25, τd = 2.0, g0 = 10.0),
    ReceptorVoltage(E_rev = 0.0, τr = 8, τd = 35.0, g0 = 0.0f0, nmda = 0.0f0),
)


"""
Richard Miles, Katalin Tóth, Attila I Gulyás, Norbert Hájos, and Tamas F Freund.  Differences between Somatic923and Dendritic Inhibition in the Hippocampus.Neuron, 16(4):815–823, April 1996. ISSN 0896-6273. doi: 10.1016/924S0896-6273(00)80101-4.
"""
MilesGabaDend = GABAergic(
    Receptor(E_rev = -75.0, τr = 4.8, τd = 29.0, g0 = 0.126),
    Receptor(E_rev = -90.0, τr = 30, τd = 100.0, g0 = 0.006),
)

MilesGabaSoma =
    GABAergic(Receptor(E_rev = -75.0, τr = 0.5, τd = 6.0, g0 = 0.265), Receptor())


"""
Renato Duarte and Abigail Morrison. Leveraging heterogeneity for neural computation with fading memory in layer 2/3808cortical microcircuits.bioRxiv, December 2017. doi: 10.1101/230821.
"""

DuarteGluSoma = Glutamatergic(
    Receptor(E_rev = 0.0, τr = 0.25, τd = 2.0, g0 = 0.73),
    ReceptorVoltage(E_rev = 0.0, nmda = 0.0f0),
)

DuarteGluDend = Glutamatergic(
    Receptor(E_rev = 0.0, τr = 0.25, τd = 2.0, g0 = 0.73),
    ReceptorVoltage(E_rev = 0.0, τr = 0.99, τd = 100.0, g0 = 0.159),
)

DuarteGabaSoma = GABAergic(
    Receptor(E_rev = -75.0, τr = 0.5, τd = 6.0, g0 = 0.265),
    Receptor(E_rev = -90.0, τr = 30, τd = 100.0, g0 = 0.006),
)


DuarteSynapsePV = let
    E_exc = 0.00       #(mV) Excitatory reversal potential
    E_gabaB = -90      #(mV) GABA_B reversal potential
    E_gabaA = -75       #(mV) GABA_A reversal potential

    gsyn_ampa = 1.040196
    # τr_ampa   = 0.087500
    τr_ampa = 0.180000
    τd_ampa = 0.700000

    gsyn_nmda = 0.002836
    τr_nmda = 0.990099
    τd_nmda = 100.000000

    gsyn_gabaA = 0.844049
    # τr_gabaA   = 0.096154
    τr_gabaA = 0.192308
    τd_gabaA = 2.500000

    gsyn_gabaB = 0.009419
    τr_gabaB = 12.725924
    τd_gabaB = 118.866124

    AMPA = Receptor(E_rev = E_exc, τr = τr_ampa, τd = τd_ampa, g0 = gsyn_ampa)
    NMDA = ReceptorVoltage(E_rev = E_exc, τr = τr_nmda, τd = τd_nmda, g0 = gsyn_nmda)
    GABAa = Receptor(E_rev = E_gabaA, τr = τr_gabaA, τd = τd_gabaA, g0 = gsyn_gabaA)
    GABAb = Receptor(E_rev = E_gabaB, τr = τr_gabaB, τd = τd_gabaB, g0 = gsyn_gabaB)

    Synapse(AMPA, NMDA, GABAa, GABAb)
end

DuarteSynapseSST = let
    E_exc = 0.00       #(mV) Excitatory reversal potential
    E_gabaA = -75       #(mV) GABA_A reversal potential
    E_gabaB = -90      #(mV) GABA_B reversal potential

    gsyn_ampa = 0.557470
    τr_ampa = 0.180000
    τd_ampa = 1.800000

    gsyn_nmda = 0.011345
    τr_nmda = 0.990099
    τd_nmda = 100.000000

    gsyn_gabaA = 0.590834
    τr_gabaA = 0.192308
    τd_gabaA = 5.000000

    gsyn_gabaB = 0.016290
    τr_gabaB = 21.198947
    τd_gabaB = 193.990036

    AMPA = Receptor(E_rev = E_exc, τr = τr_ampa, τd = τd_ampa, g0 = gsyn_ampa)
    NMDA = ReceptorVoltage(E_rev = E_exc, τr = τr_nmda, τd = τd_nmda, g0 = gsyn_nmda)
    GABAa = Receptor(E_rev = E_gabaA, τr = τr_gabaA, τd = τd_gabaA, g0 = gsyn_gabaA)
    GABAb = Receptor(E_rev = E_gabaB, τr = τr_gabaB, τd = τd_gabaB, g0 = gsyn_gabaB)
    Synapse(AMPA, NMDA, GABAa, GABAb)
end

"""
Litwin-Kumar, A., & Doiron, B. (2014). Formation and maintenance of neuronal assemblies through synaptic plasticity. Nature Communications, 5(1). https://doi.org/10.1038/ncomms6319
"""

LKDGluSoma = Glutamatergic(
    Receptor(E_rev = 0.0, τr = 1.0, τd = 6.0, g0 = 1.0),
    ReceptorVoltage(E_rev = 0.0),
)

LKDGabaSoma = GABAergic(Receptor(E_rev = -75.0, τr = 0.5, τd = 2.0, g0 = 1.0), Receptor())

export EyalGluDend, MilesGabaDend, DuarteGluSoma, MilesGabaSoma, EyalGluDend_nonmda



# ##
# LKDSynapses = SynapseModels(
#     Esyn_soma = Synapse(LKDGluSoma, LKDGabaSoma),
#     Esyn_dend = Synapse(EyalGluDend,MilesGabaDend),
#     Isyn_sst = DuarteSynapseSST,
#     Isyn_pv = Synapse(LKDGluSoma, LKDGabaSoma)
# )
#
# TripodSynapses = SynapseModels(
#     Esyn_soma=Synapse(DuarteGluSoma,MilesGabaSoma),
#     Esyn_dend=Synapse(EyalGluDend,MilesGabaDend),
#     Isyn_sst=DuarteSynapseSST,
# 	Isyn_pv=DuarteSynapsePV
# )
#
# DuarteSynapses = SynapseModels(
#     Esyn_soma=Synapse(DuarteGluSoma,DuarteGabaSoma),
#     Esyn_dend=Synapse(EyalGluDend,MilesGabaDend),
#     Isyn_sst=DuarteSynapseSST,
# 	Isyn_pv=DuarteSynapsePV
# )
