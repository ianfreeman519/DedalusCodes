from doctest import master
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

#logger.info("str") gets printed by only processor rank 0
#logger.warning("useful str") has each processor output the string and it will label which processor printed it

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
# Old resolutions:
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
    
    return solver2, np.sqrt(vals)

"""The following are parameters for the actual star:"""

Prot = 0.897673 # days - period of rotation for the actual star
f_rot = 1/Prot # 1/d - frequecy of rotation for the actual star
ell = 2 # script L
m = -1 # azimuthal wave number?
f_cor = f_obs - m*f_rot # Frame of reference shift?
om_rot = 2*np.pi*f_rot/24/60/60
om_cor = 2*np.pi*f_cor/24/60/60 # rad/s
Om = om_rot/om_cor
N2 = N2/om_cor**2
r = r/R
Br = Br/(R*om_cor)
vA = Br/np.sqrt(4*np.pi*rho)

def updateValues(newProt=Prot, newf_rot=f_rot, newell=ell, newM=m, newf_cor=f_cor, newom_cor=om_cor, newOm=Om, newN2=N2, newr=r, newBr=Br, newvA=vA):
    global Prot, f_rot, ell, m, f_cor, om_rot, om_cor, Om, N2, r, Br, vA

    Prot = 0.897673 # days - period of rotation for the actual star
    f_rot = 1/Prot # 1/d - frequecy of rotation for the actual star
    ell = 2 # script L
    m = -1 # azimuthal wave number?
    f_cor = f_obs - m*f_rot # Frame of reference shift?
    om_rot = 2*np.pi*f_rot/24/60/60
    om_cor = 2*np.pi*f_cor/24/60/60 # rad/s
    Om = om_rot/om_cor
    N2 = N2/om_cor**2
    r = r/R
    Br = br/(R*om_cor)
    vA = Br/np.sqrt(4*np.pi*rho)

def numpy_fillna(data):
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:,None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape)
    out[mask] = np.concatenate(data)
    return out

resolutionBr = int(input("Magnetic Field resolution (quick 10): "))
resolutionN2 = int(input("N2 resolution (quick 10): "))

# The final output list with dimensions of numBr by numN2 by 5, with only the first 5 real eigenvalues saved
masterOutputList = np.zeros((resolutionBr, resolutionN2, 5))
# Resetting all the output files to np.Nans, so they can be written to a file and plotted without error
masterOutputList[:] = np.nan

# creating a logorithmically spaced set of values to be tested at
brlistGAUSS = np.logspace(5.6, 6.01, num=resolutionBr)
N2list = np.logspace(np.log10(N2)-0.5, np.log10(N2)+0.5, num=resolutionN2)
vAlist = brlistGAUSS/np.sqrt(4*np.pi*rho)/(R*om_cor)

# tracking the lengths of the eigenvalues will be useful later on
lenlist = np.zeros(resolutionBr*resolutionN2)
iterationstep = -1
for j, N2 in enumerate(N2list[rank::size]):
    for i, br in enumerate(brlistGAUSS):
        iterationstep += 1
        print((iterationstep)/(resolutionBr*resolutionN2))
        N2 = N2list[rank::size][j]
        vA = vAlist[i]
        solver, kr = converged_vals(Om, R, m, vA, N2, r)
        kr = np.sort(kr.real)
        kr = kr[:5]

        masterOutputList[i, j, :len(kr)] = kr
        print(len(kr))
        lenlist[iterationstep]=len(kr)

        kr = []

saveFileName = "eigenvaluesVsN2andBrwithlengthsTEST.txt"
if os.path.exists(saveFileName):
    os.remove(saveFileName)

outfile = open(saveFileName, "w")
outfile.write("# N2(1/s)\tBr(G)\t\tvA\t\tl=1\t\tl=2\t\t=3\t\tl=4\t\tl=5\t\tNkr\n")

iterationstep = -1
for i, brs in enumerate(brlistGAUSS):
    for j, N2s in enumerate(N2list):
        iterationstep += 1
        outfile.write("{:6.4}  \t{:6.4}  \t{:6.4}  \t{:6.4}  \t".format(N2s,brs,vAlist[i],masterOutputList[i, j, 0]) \
            + "{:6.4}  \t{:6.4}  \t{:6.4}  \t{:6.4} \t".format(masterOutputList[i, j, 1],masterOutputList[i, j, 2],masterOutputList[i, j, 3],masterOutputList[i, j, 4]) \
                + str(int(lenlist[iterationstep])) + "\n")
