struct BlochSimpleWithCoils <: SimulationMethod end
const BlochSimpleWithCoills = BlochSimpleWithCoils

export BlochSimpleWithCoils, BlochSimpleWithCoills

function sim_output_dim(
    obj::Phantom, seq::Sequence, sys::Scanner, sim_method::BlochSimpleWithCoils
)
    return (sum(seq.ADC.N), 1)
end

function split_sig_per_thread(sig, i, p, sim_method::BlochSimpleWithCoils)
    return @view sig[:, :, i]
end

function initialize_spins_state(
    obj::Phantom{T}, sim_method::BlochSimpleWithCoils
) where {T<:Real}
    Nspins = length(obj)
    Mxy = zeros(T, Nspins)
    Mz = obj.ρ
    Xt = Mag{T}(Mxy, Mz)
    return Xt, obj
end

prealloc(sim_method::BlochSimpleWithCoils, backend::KA.Backend, obj::Phantom{T}, M::Mag{T}, max_block_length::Integer, groupsize) where {T<:Real} = DefaultPrealloc{T}()

function run_spin_precession!(
    p::Phantom{T},
    seq::DiscreteSequence{T},
    sig::AbstractArray{Complex{T}},
    M::Mag{T},
    sim_method::BlochSimpleWithCoils,
    groupsize,
    backend::KA.Backend,
    prealloc::PreallocResult
) where {T<:Real}
    x, y, z = get_spin_coords(p.motion, p.x, p.y, p.z, seq.t')
    Bz = x .* seq.Gx' .+ y .* seq.Gy' .+ z .* seq.Gz' .+ p.Δw ./ T(2π .* γ)
    if is_ADC_on(seq)
        ϕ = T(-2π .* γ) .* cumtrapz(seq.Δt', Bz)
    else
        ϕ = T(-2π .* γ) .* trapz(seq.Δt', Bz)
    end
    tp = cumsum(seq.Δt)
    dur = sum(seq.Δt)
    Mxy = M.xy .* exp.(-tp' ./ p.T2) .* cis.(ϕ)
    M.xy .= Mxy[:, end]
    M.z .= M.z .* exp.(-dur ./ p.T1) .+ p.ρ .* (1 .- exp.(-dur ./ p.T1))
    outflow_spin_reset!(Mxy, seq.t[2:end]', p.motion)
    outflow_spin_reset!(M, seq.t[2:end]', p.motion; replace_by=p.ρ)
    sig .= @views transpose(sum(Mxy[:, findall(seq.ADC[2:end])]; dims=1))
    return nothing
end

function run_spin_excitation!(
    p::Phantom{T},
    seq::DiscreteSequence{T},
    sig::AbstractArray{Complex{T}},
    M::Mag{T},
    sim_method::BlochSimpleWithCoils,
    groupsize,
    backend::KA.Backend,
    prealloc::PreallocResult
) where {T<:Real}
    sample = 1
    ψ_start = @view seq.ψ[1:1]
    @. M.xy = M.xy * cis(-ψ_start)
    for i in eachindex(seq.Δt)
        s = @views (
            t = seq.t[i, :], tnew = seq.t[i + 1, :], Δt = seq.Δt[i, :],
            Gx = seq.Gx[i, :], Gy = seq.Gy[i, :], Gz = seq.Gz[i, :],
            B1 = seq.B1[i, :], Δf = seq.Δf[i, :],
            ADC = any(seq.ADC[i + 1, :])
        )
        x, y, z = get_spin_coords(p.motion, p.x, p.y, p.z, s.t)
        ΔBz = p.Δw ./ T(2π .* γ) .- s.Δf ./ T(γ)
        Bz = (s.Gx .* x .+ s.Gy .* y .+ s.Gz .* z) .+ ΔBz
        B = sqrt.(abs.(s.B1) .^ 2 .+ abs.(Bz) .^ 2)
        B .+= (B .== 0) .* eps(T)
        φ = T(-2π .* γ) .* (B .* s.Δt)
        mul!(Q(φ, s.B1 ./ B, Bz ./ B), M)
        @. M.xy = M.xy * exp(-s.Δt / p.T2)
        @. M.z = M.z * exp(-s.Δt / p.T1) + p.ρ * (1 - exp(-s.Δt / p.T1))
        outflow_spin_reset!(M, s.tnew, p.motion; replace_by=p.ρ)
        if s.ADC
            acquire_signal!(sig, sample, M, sim_method)
            sample += 1
        end
    end
    ψ_end = @view seq.ψ[end:end]
    @. M.xy = M.xy * cis(ψ_end)
    return nothing
end

function acquire_signal!(sig, sample, M, sim_method::BlochSimpleWithCoils)
    sig[sample, :] .= sum(M.xy)
end
