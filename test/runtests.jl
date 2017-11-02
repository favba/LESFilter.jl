using LESFilter
using InplaceRealFFTW
using Base.Test

#=
Test function: f = sin(3x)+cos(3y)+sin(4z)
GaussianFilter(Δ,f) = exp(-2Δ^2/3) * (exp(7Δ^2/24) * (cos(3y) + sin(3x)) + sin(4z))
=#

lx = 1.0
ly = 1.0
lz = 1.0

nx = 32
ny = 32
nz = 32

x = reshape(linspace(0,lx*2π*(1-1/nx),nx),(nx,1,1))
y = reshape(linspace(0,ly*2π*(1-1/ny),ny),(1,ny,1))
z = reshape(linspace(0,lz*2π*(1-1/nz),nz),(1,1,nz))

field = @. sin(3*x) + cos(3*y) + sin(4*z) 

correct(Δ::Real) = @. exp(-2Δ^2/3) * (exp(7Δ^2/24) * (cos(3y) + sin(3x)) + sin(4z)) 
dx = x[2]
for df in 1:0.5:10
  Δ = df*dx
  @test lesfilter(field,boxdim=Δ,lengths=(lx,ly,lz)) ≈ correct(Δ)
end

field2 = PaddedArray(field)

for df in 1:0.5:10
  Δ = df*dx
  @test lesfilter!(field2,boxdim=Δ,lengths=(lx,ly,lz)) ≈ correct(Δ)
  copy!(real(field2),field)
end

rfft!(field2)
@inferred LESFilter.loopgaussian!(complex(field2),rfftfreq(nx,lx),fftfreq(ny,ly),fftfreq(nz,lz),8*dx)