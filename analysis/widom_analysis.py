#/usr/bin/env python

import sys, os, argparse, linecache, re
import io
import numpy as np
import pandas as pd
import math
import statistics as stat

R = 8.31446261815324e-3 # gas constant in kJ/mol


def widom_analysis(widomfilename, T, maxsep=5, printmu=False, printR=False):
    with open(widomfilename) as wprpfile:
        wprpfile.readline()
        widomarray = np.loadtxt(wprpfile, usecols=1)
        N = len(widomarray)
        if printmu:
            sub_N = int(N/3)
            sub_wmeans = np.array([np.mean(widomarray[:sub_N]), np.mean(widomarray[sub_N:2*sub_N]), np.mean(widomarray[2*sub_N:3*sub_N])])
            submu = -R*T*np.log(sub_wmeans) # sub-run mu' values in kJ/mol
            submu_stderr = stat.stdev(submu) # standard error of sub-run mu'
            mu = -R*T*np.log(np.mean(widomarray)) # overall estimate of mu' in kJ/mol
            mu_interval = np.array([mu-submu_stderr,mu,mu+submu_stderr])
            f_stderr = '{:.0e}'.format(submu_stderr)
            mu_precision = f_stderr[3:]
            formatted_mu_string = ('The shifted chemical potential is {:.'+mu_precision+'f}('+f_stderr[0]+') kJ/mol').format(mu)
            print(formatted_mu_string)
            if not ('-' in f_stderr or mu_precision == '00'):
                print("Error is actually ", submu_stderr)
        stepmu = -R*T*np.log(widomarray)
        stepmu_shiftmat = np.zeros((maxsep+1, N-maxsep))
        stepwvar_shiftmat = np.zeros((maxsep+1, N-maxsep))
        for i in range(maxsep+1):
            stepmu_shiftmat[i] = stepmu[i:(N-maxsep+i)]
            stepwvar_shiftmat[i] = widomarray[i:(N-maxsep+i)]
        Rmat_mu = np.corrcoef(stepmu_shiftmat)
        Rvec_mu = Rmat_mu[0]
        Rmat_wvar = np.corrcoef(stepwvar_shiftmat)
        Rvec_wvar = Rmat_wvar[0]
        df_R = pd.DataFrame({"R_mu": Rvec_mu, "R_wvar": Rvec_wvar})
        if printR:
            print(df_R.to_string())

        return df_R, stepmu, widomarray, mu_interval








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('wprp_filename')
    parser.add_argument('temperature',type=float)
    args = parser.parse_args()
    widom_analysis(widomfilename=args.wprp_filename, T=args.temperature, printmu=True, printR=True)







