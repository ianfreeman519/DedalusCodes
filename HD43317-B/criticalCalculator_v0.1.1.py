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

minN2 = (1 *2*np.pi/1000/1000)**2/om_cor**2 # microHz to rad/s
maxN2 = (3200 *2*np.pi/1000/1000)**2/om_cor**2 # microHz to rad/s
print(minN2, maxN2)

def numpy_fillna(data):
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:,None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape)
    out[mask] = np.concatenate(data)
    return out

resolutionBr = 500
resolutionN2 = 500

# The final output list with dimensions of numBr by numN2/size by 5, with only the first 5 real eigenvalues saved
masterOutputList = np.zeros((int(np.ceil(resolutionN2/size)), resolutionBr,  5))
# Resetting all the output files to np.Nans, so they can be written to a file and plotted without error
masterOutputList[:] = np.nan

# creating a logorithmically spaced set of values to be tested at
brlistGAUSS = np.logspace(4, 7, num=resolutionBr)
N2list = np.linspace(minN2, maxN2, num=resolutionN2)
vAlist = brlistGAUSS/np.sqrt(4*np.pi*rho)/(R*om_cor)

# tracking the lengths of the eigenvalues will be useful later on
lenlist = np.zeros(resolutionBr*resolutionN2)
iterationstep = -1
for i, N2 in enumerate(N2list[rank::size]):
    for j, br in enumerate(brlistGAUSS):
        iterationstep += 1
        N2 = N2list[rank::size][i]
        vA = vAlist[j]
        solver, kr = converged_vals(Om, R, m, vA, N2, r)
        kr = np.sort(kr.real)
        print(rank, len(kr))
        kr = kr[:5]

        masterOutputList[i, j, :len(kr)] = kr
        lenlist[iterationstep]=len(kr)

        kr = []

masterOutputList = np.array(MPI.COMM_WORLD.gather(masterOutputList, root=0))

if rank==0:
    saveFileName = "simulatedKrPickled.pkl"
    if os.path.exists(saveFileName):
        os.remove(saveFileName)

    oldshape = masterOutputList.shape
    newshape = (oldshape[0]*oldshape[1], oldshape[2], oldshape[3])

    # Tranposing the 0th and 1st axes so they are ordered by N2, not processor number
    masterOutputList = masterOutputList.transpose((1, 0, 2, 3))
    # Fixing the dimensions into a 3d array
    masterOutputList = masterOutputList.reshape(newshape)

    masterPickleDict = {"N2s": N2list, "Brs": brlistGAUSS, "vAs": vAlist, "krs": masterOutputList}

    outputFile = open(saveFileName, "wb")
    pickle.dump(masterPickleDict, outputFile)
    outputFile.close()


# TRY TO GET THE DATA TO BE IN PICKLE FORMAT
# TRY TO FIDDLE WITH RUNNING STUFF ON QUEST
# GET THE CRITICAL vA AS A FUNCTION OF N2
