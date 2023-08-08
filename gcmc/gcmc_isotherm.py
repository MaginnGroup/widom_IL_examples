
import pandas as pd
import mbuild as mb
import foyer
import unyt as u
import numpy as np
import math
import time
import parmed as pmd
import mosdef_cassandra as mc

T = 298.15 * u.K

mu_ideal_100box = -48.860310174012 * u.Unit('kJ/mol')

beta = 1/(u.kb*T)

N_goal = 40

cbrt_N_goal = N_goal ** (1.0/3.0)

mu_gas_5 = np.array([-49.04, -39.4, -36.4, -35.2, -35.0]) * u.Unit('kJ/mol')

P_gas = np.linspace(0.5, 15.0, 30, endpoint=True) * u.bar

P_n60 = 1.9121951616833333 * u.bar

mu_n60 = mu_gas_5[1]

mu_gas = mu_n60 + u.kb*T*np.log(P_gas/P_n60)



L = np.exp((-beta/3*(mu_gas - mu_ideal_100box)).to(''))*(100 * u.angstrom)*cbrt_N_goal

r32_struct = pmd.load_file('topology/r32.top', xyz='topology/r32.gro')
species_list = [r32_struct]

ensemble = 'gcmc'
moveset = mc.MoveSet(ensemble, species_list)
moveset.max_translate = [[14.0*u.angstrom]]
moveset.max_rotate = [[180.0*u.degree]]

eq_length = 1000000
prod_length = eq_length*10

property_list = ["nmols", "density", "pressure", "energy_total"]

def read_prp(filename):
    with open(filename) as fprp:
        fprp.readline()
        colnames = fprp.readline()[1:].strip().split()
        colunits = fprp.readline()[1:].strip().split()
        df = pd.read_csv(fprp, delim_whitespace=True, index_col=False, names=colnames)
    return df

statdf = pd.DataFrame(
        columns=['mu', 'Nmols', 'Nmols_SEM', 'Density', 'Density_SEM', 'Pressure', 'Pressure_SEM', 'Energy_Total', 'Energy_Total_SEM', 'Sim_Time'],
        index=pd.RangeIndex(len(mu_gas)))

for i, mu in enumerate(mu_gas):
    eq_name = "eq"+str(i)
    prod_name = "prod"+str(i)
    boxlength = L[i]
    cutoff = boxlength/2 - 0.1*u.angstrom
    nm_L = boxlength.to_value('nm')
    box = mb.Box([nm_L, nm_L, nm_L])
    box_list = [box]
    system = mc.System(box_list, species_list, mols_to_add=[[N_goal]])
    starttime = time.time()
    mc.run(
            system=system,
            moveset=moveset,
            run_type="equilibration",
            run_length=eq_length,
            temperature=T,
            chemical_potentials=[mu],
            prop_freq=1000,
            coord_freq=10000,
            properties=property_list,
            vdw_cutoff=cutoff,
            charge_cutoff=cutoff,
            run_name=eq_name
    )
    mc.restart(
            restart_from=eq_name,
            run_name=prod_name,
            run_type="production",
            total_run_length=prod_length
    )
    endtime = time.time()
    prpdf = read_prp(prod_name+".out.prp")
    prp_mean = prpdf.mean(axis=0)
    prp_sem = prpdf.sem(axis=0)
    statdf['mu'][i] = mu.to_value('kJ/mol')
    statdf['Pressure'][i] = prp_mean['Pressure']
    statdf['Pressure_SEM'][i] = prp_sem['Pressure']
    statdf['Nmols'][i] = prp_mean['Nmols']
    statdf['Nmols_SEM'][i] = prp_sem['Nmols']
    statdf['Density'][i] = prp_mean['Density']
    statdf['Density_SEM'][i] = prp_sem['Density']
    statdf['Energy_Total'][i] = prp_mean['Energy_Total']
    statdf['Energy_Total_SEM'][i] = prp_sem['Energy_Total']
    statdf['Sim_Time'][i] = endtime-starttime

statdf.to_csv('property_data.csv', index=False)

