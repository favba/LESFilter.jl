__precompile__()
module LESFilter

using InplaceRealFFT

export rfftfreq, fftfreq, rfftfreqn, fftfreqn, lesfilter!, lesfilter

function rfftfreq(n::Integer,s::Real) 
  d = 2π*s
  Float64[(n/2 - i)/d for i = n/2:-1:0]
end

function fftfreq(n::Integer,s::Real)
  d = 2π*s
  if iseven(n)
    return vcat(Float64[(n/2 - i)/(d) for i = n/2:-1:1],Float64[-i/(d) for i = n/2:-1:1])
  else return vcat(Float64[(n/2 - i)/(d) for i = n/2:-1:0],Float64[-i/(d) for i = (n-1)/2:-1:1])
  end
end

function rfftfreqn(n::Integer,d::Real)
  Float64[(n/2 - i)/d for i = n/2:-1:0]
end

function fftfreqn(n::Integer,d::Real)
  if iseven(n)
    return vcat(Float64[(n/2 - i)/(d) for i = n/2:-1:1],Float64[-i/(d) for i = n/2:-1:1])
  else return vcat(Float64[(n/2 - i)/(d) for i = n/2:-1:0],Float64[-i/(d) for i = (n-1)/2:-1:1])
  end
end

function lesfilter(field::AbstractArray{<:Real,3} , fil::String, boxdim::Real,lengths::Tuple{Real,Real,Real})
  nx,ny,nz = size(field)
  xs,ys,zs = lengths

  fieldhat = rfft(field)

  kx2 = rfftfreq(nx,xs).^2
  ky2 = fftfreq(ny,ys).^2
  kz2 = fftfreq(nz,zs).^2

  loopgaussian!(fieldhat,kx2,ky2,kz2,boxdim)

  return irfft(fieldhat,nx)
end

function loopgaussian!(fieldhat::AbstractArray{<:Complex,3},kx2::AbstractVector,ky2::AbstractVector,kz2::AbstractVector,boxdim::Real)
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

function loopcutoff!(fieldhat::AbstractArray{<:Complex,3},kx2::AbstractVector,ky2::AbstractVector,kz2::AbstractVector,boxdim::Real)
  aux = ((1/2boxdim)^2)
  Threads.@threads for k = 1:length(kz2)
    for j = 1:length(ky2)
      @simd for i = 1:length(kx2)
        @inbounds fieldhat[i,j,k] = fieldhat[i,j,k]*((kx2[i]+ky2[j]+kz2[k]) < aux)
      end
    end
  end
  nothing
end

function loopbox!(fieldhat::AbstractArray{<:Complex,3},kx2::AbstractVector,ky2::AbstractVector,kz2::AbstractVector,boxdim::Real)
  aux = π*boxdim
  a = fieldhat[1]
  Threads.@threads for k = 1:length(kz2)
    for j = 1:length(ky2)
      @fastmath @simd for i = 1:length(kx2)
        mk = sqrt(kx2[i]+ky2[j]+kz2[k])
        @inbounds fieldhat[i,j,k] = fieldhat[i,j,k]*sinpi(mk*boxdim)/(mk*aux)
      end
    end
  end
  fieldhat[1] = a
  nothing
end

function loopanigaussian!(fieldhat::AbstractArray{<:Complex,3},kx2::AbstractVector,ky2::AbstractVector,kz2::AbstractVector,boxdimxy::Real,boxdimz::Real)
  auxxy = -((π*boxdimxy)^2)/6
  auxz = -((π*boxdimz)^2)/6
  Threads.@threads for k = 1:length(kz2)
    for j = 1:length(ky2)
      @simd for i = 1:length(kx2)
        @inbounds fieldhat[i,j,k] = fieldhat[i,j,k]*exp((kx2[i]+ky2[j])*auxxy + kz2[k]*auxz)
      end
    end
  end
  nothing
end

function lesfilter(field::AbstractPaddedArray{<:Real,3} , fil::String, boxdim::Real,lengths::NTuple{3,Real})
  newfield = copy(field)

  lesfilter!(newfield,fil=fil,boxdim=boxdim,lengths=lengths)

  return newfield
end


function lesfilter!(field::AbstractPaddedArray{<:Real,3}, fil::String, boxdim::Real, lengths::NTuple{3,Real})
  nx,ny,nz = size(real(field))
  xs,ys,zs = lengths

  fieldhat = complex(rfft!(field))

  kx2 = rfftfreq(nx,xs).^2
  ky2 = fftfreq(ny,ys).^2
  kz2 = fftfreq(nz,zs).^2

  if fil == "G"
    loopgaussian!(fieldhat,kx2,ky2,kz2,boxdim)
  elseif fil == "C"
    loopcutoff!(fieldhat,kx2,ky2,kz2,boxdim)
  elseif fil == "B"
    loopbox!(fieldhat,kx2,ky2,kz2,boxdim)
  end

  return irfft!(field)
end

function lesfilter!(field::AbstractPaddedArray{<:Real,3}, fil::String, boxdims::NTuple{2,Real}, lengths::NTuple{3,Real})
  nx,ny,nz = size(real(field))
  xs,ys,zs = lengths

  fieldhat = complex(rfft!(field))

  kx2 = rfftfreq(nx,xs).^2
  ky2 = fftfreq(ny,ys).^2
  kz2 = fftfreq(nz,zs).^2

  if fil == "G"
    loopanigaussian!(fieldhat,kx2,ky2,kz2,boxdims[1],boxdims[2])
  elseif fil == "C"
    #loopanicutoff!(fieldhat,kx2,ky2,kz2,boxdim,boxdimz)
  elseif fil == "B"
    #loopanibox!(fieldhat,kx2,ky2,kz2,boxdim,boxdimz)
  end
  return irfft!(field)
end

end # module
