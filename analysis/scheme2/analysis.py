import numpy as np
import pandas as pd
import unyt as u
import io, os, shutil
import random
from pathlib import Path

n_solute_list = [0, 60, 260, 500, 800]

cation = "bmim"
anion = "pf6"

pre_seed_fulltraj = ""
pre_seed_top500fromhalf = ""

nframes = 50000
nreps = 10
T = 298.15 * u.K
beta = 1/(u.kb*T)
mu_ig_100angstroms = -48.860310174012 * u.Unit('kJ/mol')

def get_thermo_data(logpath,skiplines=0):
    logpath = Path(logpath)
    with logpath.open() as logfile, io.StringIO() as thermobuffer:
        linestr = ' '
        while linestr and not ('Per MPI rank memory allocation' in linestr):
            linestr = logfile.readline()
        linestr = logfile.readline()
        for i in range(skiplines):
            logfile.readline()
        while not ('Loop time of' in linestr):
            thermobuffer.write(linestr)
            linestr = logfile.readline()
        thermobuffer.seek(0)
        thermodf = pd.read_csv(thermobuffer, delim_whitespace=True)
    return thermodf


for n_solute in n_solute_list:
    concpath = Path('../../n'+str(n_solute))
    ltraj_path = Path(concpath, 'lmp_1_prod.lammpstrj')
    schemepath = Path(concpath, 'scheme2')
    modepath = Path(schemepath, 'nonchunk')
    conc_analysis_path = Path('./n'+str(n_solute))
    log_path = Path(concpath,'lmp_1_prod.log')
    thermoprops = get_thermo_data(log_path,skiplines=1)
    V = thermoprops.Volume.mean() * u.angstrom ** 3
    V_empty = (100 * u.angstrom) ** 3
    density = (300+n_solute) / V
    mu_ig = mu_ig_100angstroms + u.kb*T*np.log((n_solute+1)*V_empty/V) 
    if not conc_analysis_path.is_dir():
        os.mkdir(conc_analysis_path)
    nlist = [300, 300]
    if n_solute:
        nlist.append(n_solute)
    fulltraj_wprp2_3d = np.dstack([np.loadtxt(Path(modepath, 'reps/rep'+str(irep+1),'fulltraj/cas_widom.out.spec3.wprp2')) for irep in range(nreps)])
    fulltraj_wprp3_3d = np.cumsum(fulltraj_wprp2_3d,1)/np.reshape(np.arange(100)+1, (1,100,1))
    top500_wprp2_3d = np.dstack([np.loadtxt(Path(modepath, 'reps/rep'+str(irep+1),'top500fromhalf/cas_widom.out.spec3.wprp2')) for irep in range(nreps)])
    top500_wprp3_3d = np.cumsum(top500_wprp2_3d,1)/np.reshape(np.arange(100)+1, (1,100,1))
    fulltraj_rmu = -u.kb*T*np.log(np.mean(fulltraj_wprp3_3d,0).reshape((100,nreps)))
    fulltraj_rmu_ex = fulltraj_rmu - mu_ig
    rmu_ex_combined = -u.kb*T*np.log(np.mean(fulltraj_wprp3_3d,(0,2))) - mu_ig
    fulltraj_rH = (u.kb*T*density*np.exp((beta*fulltraj_rmu_ex).to(''))).to('MPa')
    H_convergence_combined = (u.kb*T*density*np.exp((beta*rmu_ex_combined).to(''))).to('MPa')
    H_convergence_std = np.std(fulltraj_rH, axis=1)
    fulltraj_wprp = np.column_stack([np.loadtxt(Path(modepath, 'reps/rep'+str(irep+1),'fulltraj/cas_widom.out.spec3.wprp'), usecols=1, skiprows=1) for irep in range(nreps)])
    np.savetxt(Path(conc_analysis_path,'wprp.txt'),fulltraj_wprp,fmt='%.12f')
    np.savetxt(Path(conc_analysis_path,'slope_convergence_multirep.txt'),fulltraj_rH,fmt='%.6f')
    np.savetxt(Path(conc_analysis_path,'slope_convergence.txt'),H_convergence_combined,fmt='%.6f')
    np.savetxt(Path(conc_analysis_path,'slope_convergence_std.txt'),H_convergence_std,fmt='%.6f')
    np.savetxt(Path(conc_analysis_path,'rmu_ex.txt'), rmu_ex_combined.to('kJ/mol'), fmt='%.6f')
    np.savetxt(Path(conc_analysis_path,'rmu_ex_multirep.txt'), fulltraj_rmu_ex.to('kJ/mol'), fmt='%.6f')
    np.savetxt(Path(conc_analysis_path,'rmu.txt'), (rmu_ex_combined+mu_ig).to('kJ/mol'), fmt='%.6f')
    np.savetxt(Path(conc_analysis_path,'rmu_multirep.txt'), fulltraj_rmu.to('kJ/mol'), fmt='%.6f')
    if not n_solute:
        np.savetxt('H_convergence_multirep.txt',fulltraj_rH,fmt='%.6f')
        np.savetxt('H_convergence.txt', H_convergence_combined,fmt='%.6f')
        np.savetxt('H_convergence_std.txt', H_convergence_std, fmt='%.6f')
    for irep in range(nreps):
        rep_path = Path(modepath, 'reps', 'rep'+str(irep+1))
        fulltraj_path = Path(rep_path, 'fulltraj')
        top500_path = Path(rep_path, 'top500fromhalf')
        rep_analysis_path = Path(rep_path, 'analysis')
        if not rep_analysis_path.is_dir():
            os.mkdir(rep_analysis_path)
        fulltraj_wprp2 = Path(fulltraj_path, 'cas_widom.out.spec3.wprp2')
        fulltraj_wprp3 = Path(fulltraj_path, 'cas_widom.out.spec3.wprp3')
        top500_wprp2 = Path(top500_path, 'cas_widom.out.spec3.wprp2')
        top500_wprp3 = Path(top500_path, 'cas_widom.out.spec3.wprp3')
        np.savetxt(fulltraj_wprp3, fulltraj_wprp3_3d[:,:,irep])
        np.savetxt(top500_wprp3, top500_wprp3_3d[:,:,irep])
        #top500_frames = np.loadtxt(Path(top500_path, 'framelist.txt'), dtype=int)
        #top500_frames_sorted = np.sort(top500_frames)
        #isampled_wvar = np.loadtxt(fulltraj_wprp3, usecols=49)
        #top500_midway_wvar = np.loadtxt(top500_wprp3, usecols=49)
        #isampled_wvar[top500_frames_sorted] = (isampled_wvar[top500_frames_sorted] + 100*top500_midway_wvar)/101 # probably not mathematically valid


