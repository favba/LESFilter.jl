__precompile__()
module LESFilter

using InplaceRealFFTW, StaticArrays

export rfftfreq, fftfreq, rfftfreqn, fftfreqn, lesfilter!, lesfilter

function rfftfreq(n::Integer,s::Real) 
  d = 2π*s
  SVector{n÷2 + 1,Float64}(Float64[(n/2 - i)/d for i = n/2:-1:0])
end

function fftfreq(n::Integer,s::Real)
  d = 2π*s
  if iseven(n)
    return SVector{n,Float64}(vcat([(n/2 - i)/(d) for i = n/2:-1:1],[-i/(d) for i = n/2:-1:1]))
  else return SVector{n,Float64}(vcat([(n/2 - i)/(d) for i = n/2:-1:0],[-i/(d) for i = (n-1)/2:-1:1]))
  end
end

function rfftfreqn(n::Integer,d::Real)
  SVector{n÷2 + 1,Float64}(Float64[(n/2 - i)/d for i = n/2:-1:0])
end

function fftfreqn(n::Integer,d::Real)
  if iseven(n)
    return SVector{n,Float64}(vcat([(n/2 - i)/(d) for i = n/2:-1:1],[-i/(d) for i = n/2:-1:1]))
  else return SVector{n,Float64}(vcat([(n/2 - i)/(d) for i = n/2:-1:0],[-i/(d) for i = (n-1)/2:-1:1]))
  end
end

function lesfilter(field::A ; fil::String="G", boxdim::Float64=nothing,lengths::Tuple{Real,Real,Real}=nothing) where A<:Union{Array{Float64,3},Array{Float32,3}}
  nx,ny,nz = size(field)
  xs,ys,zs = lengths

  fieldhat = rfft(field)

  kx2 = rfftfreq(nx,xs).^2
  ky2 = fftfreq(ny,ys).^2
  kz2 = fftfreq(nz,zs).^2

  loopgaussian!(fieldhat,kx2,ky2,kz2,boxdim)

  return irfft(fieldhat,nx)
end

function loopgaussian!(fieldhat::AbstractArray{Complex128,3},kx2::AbstractVector,ky2::AbstractVector,kz2::AbstractVector,boxdim::Real)
  aux = -((π*boxdim)^2)/6
  Threads.@threads for k = 1:length(kz2)
    for j = 1:length(ky2)
      @simd for i = 1:length(kx2)
        @inbounds fieldhat[i,j,k] = fieldhat[i,j,k]*exp((kx2[i]+ky2[j]+kz2[k])*aux)
      end
    end
  end
  nothing
end

function lesfilter(field::AbstractPaddedArray{Float64,3,false} ; fil::String="G", boxdim::Real=nothing,lengths::NTuple{3,Real}=nothing)
  nx,ny,nz = size(real(field))
  xs,ys,zs = lengths
  newfield = copy(field)
  fieldhat = complex(rfft!(newfield))

  kx2 = rfftfreq(nx,xs).^2
  ky2 = fftfreq(ny,ys).^2
  kz2 = fftfreq(nz,zs).^2

  loopgaussian!(fieldhat,kx2,ky2,kz2,boxdim)

  irfft!(newfield)
  return newfield
end


function lesfilter!(field::AbstractPaddedArray{Float64,3,false} ; fil::String="G", boxdim::Real=nothing,lengths::NTuple{3,Real}=nothing)
  nx,ny,nz = size(real(field))
  xs,ys,zs = lengths

  fieldhat = complex(rfft!(field))

  kx2 = rfftfreq(nx,xs).^2
  ky2 = fftfreq(ny,ys).^2
  kz2 = fftfreq(nz,zs).^2

  loopgaussian!(fieldhat,kx2,ky2,kz2,boxdim)

  return irfft!(field)
end


end # module
