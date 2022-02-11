## CMSG units
const metre = 1e2
const meter = metre
const cm = metre / 1e2 # 1
const mm = metre / 1e3
const um = metre / 1e6
const nm = metre / 1e9

const second = 1e3
const ms = second / 1e3 # 1
const Hz = 1 / second

const voltage = 1e3
const mV = voltage / 1e3 #1

const ampere = 1e6
const mA = ampere / 1e3
const uA = ampere / 1e6 #1
const nA = ampere / 1e9
const pA = ampere / 1e12

const farad=1e12
const uF = farad/ 1e9
const pF = farad/1e12 ## 1
const ufarad = uF

const siemens = 1e9
const mS = siemens / 1e3
const msiemens = mS
const nS = siemens / 1e9 #1
const nsiemens = nS

const ohm = 10e-9
const Ω = ohm 
const MΩ = 10e6ohm
const GΩ = 10e9ohm


macro load_units()
    exs = map((:metre, :meter, :cm, :mm, :um, :nm, :second, :ms, :Hz, :voltage, :mV, :ampere,
        :mA, :uA, :nA, :farad, :uF, :pF, :siemens, :mS, :nS, 
        :ohm, :Ω,:MΩ,:GΩ, :ohm)) do s
        :($s = getfield($@__MODULE__, $(QuoteNode(s))))
    end
    ex = Expr(:block, exs...)
    esc(ex)
end
