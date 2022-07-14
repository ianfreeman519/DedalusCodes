import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
from mpi4py import MPI
from tomso import gyre
import pickle
import time
import os

rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

weak = True
file = 'GYRE/m5800_z014_ov004_profile_at_xc540_l2m-1_frequencies.ad'
mode_data = gyre.load_summary(file)

if weak:
    f_obs = mode_data['Refreq'][-14]
    logger.info(f_obs)
    name = 'eigenfunctions_weak.pk1'
else:
    f_obs = mode_data['Refreq'][-15]
    logger.info(f_obs)
    name = 'eigenfunctions_strong.pk1'

# Getting data from the entire star using np.loadtxt()
data = np.loadtxt('best.data.GYRE')
r=data[:,1]
rho=data[:,6]
N2=data[:,8]
R=r[-1]
Br=3e5
print('Radius:', r)

# Getting the data values for r=0.18R
indexr18R = 1330 # the index when r~0.18R
r=r[indexr18R]
rho=rho[indexr18R]
N2=N2[indexr18R]

# Parameters - again, not too sure where these come in...
Nphi = 4
dtype = np.complex128

# Bases
coords = d3.S2Coordinates('phi', 'theta')
dist = d3.Distributor(coords, dtype=dtype, comm=MPI.COMM_SELF)

# Resolutions at which the problem will be solved:
# res1 = 128
# res2 = 192
res1 = 64
res2 = 96


basis_lres = d3.SphereBasis(coords, (Nphi, res1), radius=1, dtype=dtype)
basis_hres = d3.SphereBasis(coords, (Nphi, res2), radius=1, dtype=dtype)

phi, theta = basis_hres.local_grids()

# Substitutions
zcross = lambda A: d3.MulCosine(d3.skew(A))
C2 = lambda A: d3.MulCosine(d3.MulCosine(A))


def solve(basis, N2, Br, Om, r, m):
    print(N2, Br, Om, r, m)

    zcross = lambda A: d3.MulCosine(d3.skew(A))
    C2 = lambda A: d3.MulCosine(d3.MulCosine(A))
    kr2 = dist.Field(name='kr2')

    u = dist.VectorField(coords, name='u', bases=basis)
    ur = dist.Field(name='ur', bases=basis)
    p = dist.Field(name='p', bases=basis)

    problem = d3.EVP([ur, u, p], eigenvalue=kr2, namespace=locals())
    problem.add_equation("N2*ur + p = 0")
    problem.add_equation("u + 1j*2*Om*zcross(u) + 1j*grad(p)/r - kr2*Br**2*C2(u) = 0")
    problem.add_equation("div(u)/r + 1j * kr2*ur = 0")

    # Solve
    solver = problem.build_solver()
    for sp in solver.subproblems:
        if sp.group[0] == m:
            solver.solve_dense(sp)

    vals = solver.eigenvalues
    vecs = solver.eigenvectors

    bad = (np.abs(vals) > 1e9)
    vals[bad] = np.nan
    vecs = vecs[:, np.isfinite(vals)]
    vals = vals[np.isfinite(vals)]

    vecs = vecs[:, np.abs(np.imag(vals)) < 10]
    vals = vals[np.abs(np.imag(vals)) < 10]
    vecs = vecs[:, vals.real > 0]
    vals = vals[vals.real > 0]

    i = np.argsort(-np.sqrt(vals).real)

    solver.eigenvalues, solver.eigenvectors = vals[i], vecs[:,i]
    return solver, vals[i]

def converged_vals(Om, R, m, Br, N2, r):

    solver1, vals1 = solve(basis_lres, N2, Br, Om, r, m)
    solver2, vals2 = solve(basis_hres, N2, Br, Om, r, m)

    vals = []
    for val in vals1:
        if np.min(np.abs(val-vals2))/np.abs(val) < 1e-7:
            vals.append(val)
    vals = np.array(vals)
    
    return solver2, np.sqrt(vals)


"""The following are parameters for the actual star:"""


Prot = 0.897673 # days - period of rotation for the actual star
f_rot = 1/Prot # 1/d - frequecy of rotation for the actual star
ell = 2 # script L
m = -1 # azimuthal number?
f_cor = f_obs - m*f_rot # Frame of reference shift?
om_rot = 2*np.pi*f_rot/24/60/60
om_cor = 2*np.pi*f_cor/24/60/60 # rad/s
Om = om_rot/om_cor
N2 = N2/om_cor**2
r = r/R


masterkr_list = []
# brlist = np.logspace(4.8, 5.8, num=20)
brlist = np.logspace(5.5, 6.2, num=10)

krMIN = 0
krMAX = 0
bMAX = 0
vAMAX = 0
vA_list = []

for i, br in enumerate(brlist):
    print(i/len(brlist))
    print("BR TP 1:", br, i)
    BrRIGHT = br/(R*om_cor)
    vA = BrRIGHT/np.sqrt(4*np.pi*rho)
    vA_list.append(vA)
    print("vA TP 2:", vA, i)
    print("BR TP 3:", BrRIGHT, i)
    time.sleep(3)
# logger.info(f_obs)
# logger.info(f_cor)

    kr_list = []

    solver, kr = converged_vals(Om, R, m, vA, N2, r)
    masterkr_list.append(kr)

    print(len(kr))

    if len(kr) == 0 and bMAX == 0:
        print("SETTING BRMAX:", br, len(kr))
        bMAX = br
        vAMAX = vA
        break

    for kri in range(len(kr)):
        if kr[kri] > krMAX:
            krMAX = kr[kri]
        if kr[kri] < krMIN:
            krMIN = kr[kri]

    kr = []

for x, y in zip(np.real(brlist), np.real(masterkr_list)):
    plt.scatter([x]*len(y), y, color="black", label="$k_r$")

plt.plot([bMAX, bMAX], [krMIN, krMAX],label="Maximum $B_r$",color="red")
# plt.plot([vAMAX, vAMAX], [krMIN, krMAX],color="red")
saveFileName = "test2.png"
plt.title('$k_r$ vs $B_r$: $B_{r,max}$ = ' + str(bMAX))
plt.xlabel("Magnetic Field Strength (G)")
plt.ylabel("$k_r$")

if os.path.exists(saveFileName):
      os.remove(saveFileName)

plt.savefig(saveFileName)

plt.show()

