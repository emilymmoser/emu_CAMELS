import numpy as np
import emcee
import matplotlib.pyplot as plt
import time
from chainconsumer import ChainConsumer
import sys
from multiprocessing import Pool
from getdist import plots, MCSamples
import os
os.environ["OMP_NUM_THREADS"]="1"
import test_mcmc_functions_add_2h as mf


z=0.54
suite=sys.argv[1]
prof=sys.argv[2]
run=sys.argv[3]

identity='test_add_2h'

home_ACT_data='/home/cemoser/Repositories/emu_CAMELS/ACT_comparison/LH_set/'
home_mop='/home/cemoser/Repositories/emu_CAMELS/mopc_profiles/'
home_mopc='/home/cemoser/Projection_Codes/Mop-c-GT-copy/'

thta_act,proj_data,cov,cov_diag=mf.load_ACT_data(home_ACT_data,prof)
#these are in muK*sr


ndim=5
theta0=[1.0,1.0,1.0,1.0,1.0] #now include theta2h at the end

nwalkers = 300
iter=1000
print("run with walkers ",nwalkers, " iter ",iter )

samples,x,y,emulator=mf.build_emulator(home_mop,suite,prof)

pos = [theta0 + 5e-3*np.random.randn(ndim) for i in range(nwalkers)]
with Pool() as pool:
    prob_model=mf.lnprob
    sampler = emcee.EnsembleSampler(nwalkers, ndim, prob_model, args=(thta_act,proj_data,cov,x,prof,emulator),pool=pool)
    start=time.time()
    sampler.run_mcmc(pos, iter)
    end=time.time()
    time_elapsed=end-start
    print ("time elapsed for mcmc", time_elapsed/60., "minutes")

    samples = sampler.chain[:, :, :].reshape((-1, ndim))
    np.savetxt('./chains/samples_total_'+suite+'_'+prof+'_run'+run+'_'+identity+'.txt',samples)
    samp_reshaped = sampler.chain[:, :, :].reshape((-1, ndim))
    cut=300
    
    samples_burn = sampler.chain[:, cut:, :].reshape((-1, ndim))
    np.savetxt('./chains/samples_burn_'+suite+'_'+prof+'_run'+run+'_'+identity+'.txt',samples_burn)
    print("#Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

    #logprob
    logprob=sampler.get_log_prob()
    np.savetxt('./chains/logprob_total_'+suite+'_'+prof+'_run'+run+'_'+identity+'.txt',logprob)
    logprob_burn=logprob[cut:,:]
    np.savetxt('./chains/logprob_burn_'+suite+'_'+prof+'_run'+run+'_'+identity+'.txt',logprob_burn)
    logprob_burn_reshape=logprob_burn.reshape(-1)
    logprob_max_index=list(logprob_burn_reshape).index(max(logprob_burn_reshape))
    sample_best=samples_burn[logprob_max_index,:]
    print("sample_best (max logprob)",sample_best)


    labels=["ASN1","AAGN1","ASN2","AAGN2","theta2h"]
    labels_getdist=[r'A_{SN1}',r'A_{AGN1}',r'A_{SN2}',r'A_{AGN2}',r'A_{2h}']
    asn1_mcmc, agn1_mcmc, asn2_mcmc,agn2_mcmc,theta2h_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples_burn, [16, 50, 84],axis=0)))
    theta_fit=[asn1_mcmc[0],agn1_mcmc[0],asn2_mcmc[0],agn2_mcmc[0],theta2h_mcmc[0]]
    
    #print out latex format
    for i in range(ndim):
        mcmc = np.percentile(samples_burn[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.2f}_{{-{1:.2f}}}^{{+{2:.2f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        latex = "${0:.2f}_{{-{1:.2f}}}^{{+{2:.2f}}}$"
        latex = latex.format(mcmc[1], q[0], q[1], labels[i])
        print(latex)

    for i in range(ndim):
        chain=samples_burn[:,i]
        c=ChainConsumer()
        c.add_chain(chain.T,walkers=nwalkers)
        gelman_rubin_converged=c.diagnostic.gelman_rubin(threshold=0.1)
        print("gelman-rubin:",gelman_rubin_converged)
    
    end2=time.time()
    total_time=end2-start
    print("total time",total_time/60.," minutes ",total_time/3600., " hours")
