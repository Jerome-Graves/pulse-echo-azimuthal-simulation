import numpy as np
rng=np.random.default_rng(1)
N=60
w=rng.normal(size=N); W=np.fft.fft(w); k=np.fft.fftfreq(N,1/N)
x=np.real(np.fft.ifft(W/(1+np.abs(k))))
x=x-x.mean()
P=np.abs(np.fft.fft(x))**2
c=np.real(np.fft.ifft(P))/N     # circular autocovariance
rho=c/c[0]
print("sum of circular ACF over ALL lags 0..N-1 :", rho.sum())
print("(theory: exactly 0 for a mean-removed circular series)")
# integral-scale ESS truncated at first zero crossing
M=1
while M<N//2 and rho[M]>0: M+=1
ess=N/(1+2*sum((1-j/N)*rho[j] for j in range(1,M)))
print("first zero crossing at lag", M, "-> N_eff =", round(ess,1))
# what happens if you DON'T truncate (sum lags 1..N-1)
den=1+2*sum((1-j/N)*rho[j] for j in range(1,N))
print("untruncated denominator:", round(den,3), "-> N_eff =", round(N/den,1))
