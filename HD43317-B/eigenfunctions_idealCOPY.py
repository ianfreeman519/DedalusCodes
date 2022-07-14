import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
from mpi4py import MPI
from tomso import gyre
import pickle
import time

rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

weak = True
file = 'GYRE/m5800_z014_ov004_profile_at_xc540_l2m-1_frequencies.ad'
mode_data = gyre.load_summary(file)
if weak: 
    f_obs = mode_data['Refreq'][-14]
    logger.info(f_obs)
    name = 'eigenfunctions_weak.pkl'
else:
    f_obs = mode_data['Refreq'][-15]
    logger.info(f_obs)
    name = 'eigenfunctions_strong.pkl'

# WHOLE STAR DATA
data = np.loadtxt('best.data.GYRE')
r = data[:,1]
rho = data[:,6]
N2 = data[:,8] # (rad/s)^2
R = r[-1]
Br = 1312*(R/r)**3 # G

# vA=Br/Sqrt(4pi*rho)

# STAR DATA AROUND r=0.18R
print(r[1330]/R)
r = r[1325:1330]
rho = rho[1325:1330]
N2 = N2[1325:1330]
Br = 5e5
time.sleep(10)


# only want to look at r=0.18*R !!
# What Br at that point (in G) gives interaction with waves??
# best to just pick a constant at somewhere around 5e5

i_mid = np.argmin(np.abs(r-R/2))
i_inner = np.where(N2[:i_mid]>2e-6)[0][0]
i_outer = np.where(N2[:i_mid]>2e-6)[0][-1]

r_inner = r[i_inner]
r_outer = r[i_outer]

i_spike = np.argmin(np.abs(r - (r_inner+r_outer)/2))
r_spike = r[i_spike]

"""What is this^^^??? i_spike... r_spike... it looks to me like its the minimum of the average radius? """

i_min = np.argmin(N2[i_spike:i_mid]) + i_spike
i_max = np.argmax(N2[:i_spike])

r_list1 = np.linspace(r[i_max], r[i_min], num=20, endpoint=False)
r_list2 = np.linspace(r[i_min], r[i_mid], num=100, endpoint=True)
r_list = np.concatenate((r_list1, r_list2))
N2_list = np.interp(r_list, r, N2)
rho_list = np.interp(r_list, r, rho)

# forget abou this ^^^
# instead you need to find N2 and rho at r=0.18*R


Bcrit = 4.68e5 # G
# Br_list really contains the Alfven velocity vA
# instead of Bcrit*(r...)**3, just pick a magnetic field strength
Br_list = Bcrit*(r_spike/r_list)**3/np.sqrt(4*np.pi*rho_list) # radial alfven velocity

# For the calculation, we need vA = br/sqrt(4 pi rho)

# Parameters
Nphi = 4
dtype = np.complex128

# Bases
coords = d3.S2Coordinates('phi', 'theta')
dist = d3.Distributor(coords, dtype=dtype, comm=MPI.COMM_SELF)

#res1 = 192
#res2 = 256
res1 = 128
res2 = 192

basis_lres = d3.SphereBasis(coords, (Nphi, res1), radius=1, dtype=dtype)
basis_hres = d3.SphereBasis(coords, (Nphi, res2), radius=1, dtype=dtype)
phi, theta = basis_hres.local_grids()

# Substitutions
zcross = lambda A: d3.MulCosine(d3.skew(A))
C2 = lambda A: d3.MulCosine(d3.MulCosine(A))

# Br should really be Alfven velocity
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
    problem.add_equation("div(u)/r + 1j*kr2*ur = 0")

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

    i = np.argsort(np.sqrt(vals).real)
    solver.eigenvalues, solver.eigenvectors = vals[i], vecs[:, i]

    return solver, vals

def converged_vals(i, Om, R, m, Br=None, N2=None, r=None):
    
    if Br is None:
        Br = Br_list[i]/(R*om_cor)
        N2 = N2_list[i]
        r = r_list[i]/R

    # everything has to be dimensionless
    solver1, vals1 = solve(basis_lres, N2, Br, Om, r, m)
    solver2, vals2 = solve(basis_hres, N2, Br, Om, r, m)

    vals = []
    for val in vals1:
        if np.min(np.abs(val - vals2))/np.abs(val) < 1e-7:
            vals.append(val) 
    vals = np.array(vals)

    return solver2, vals

# Figuring out parameters
Prot = 0.897673  #d
f_rot = 1/Prot
ell = 2
m = -1
f_cor = f_obs - m*f_rot
# f_cor = f_obs
om_rot = 2*np.pi*f_rot/24/60/60
om_cor = 2*np.pi*f_cor/24/60/60 # rad/sec
Om = om_rot/om_cor
N2_list = N2_list/om_cor**2

logger.info(f_obs)
logger.info(f_cor)

kr_list = []
r_local = []
uph_list = []
for j in range(rank, len(r_list), size):
    r_local.append(r_list[j]/R)

    # just calculate eigenvalues as a function of Br
    solver, vals = converged_vals(j, Om, R, m)
    # print eigenvalues to see what they look like
    # what changes when magnetic field strength moves above or below 5e5 G
    print('TESTING: ', vals)

    if len(vals) > 1:

        kr = np.sqrt(vals)
    
        kr_list.append(kr)
        for sp in solver.subproblems:
            if sp.group[0] == m:
                break
        logger.info(np.sqrt(vals))

    else:
        logger.warning('only one eigenvalue')
        logger.warning(i)

#kr_list_list = MPI.COMM_WORLD.gather(kr_list, root=0)
#r_list_list = MPI.COMM_WORLD.gather(r_local, root=0)
#uph_list_list = MPI.COMM_WORLD.gather(uph_list, root=0)
#
#if rank == 0:
#    uph_list = []
#    kr_list = []
#    r_list = []
#    
#    for r_l, kr_l, uph_l in zip(r_list_list, kr_list_list, uph_list_list):
#        for kr, r, uph in zip(kr_l, r_l, uph_l):
#            kr_list.append(kr)
#            r_list.append(r)
#            uph_list.append(uph)
#
#    r_list = np.array(r_list)
#    i = np.argsort(r_list)
#    r_list = r_list[i]
#    kr_list = np.array(kr_list)[i]
#    uph_list = np.array(uph_list)[i]
#
#    plot = False
#
#    if plot:
#        for kr, x in zip(kr_list, r_list):
#            kr = np.array(kr)
#            print(x, kr)
#            if kr.size > 0:
#                plt.scatter(kr.real, [x]*kr.size)
#    
#        plt.xlabel('kr')
#        plt.ylabel('r')
#        plt.savefig('kr.png', dpi=150)
#
#    data = {'th': theta, 'r': r_list, 'kr':kr_list, 'uph':uph_list}
#    pickle.dump(data, open(name, 'wb'))
#
