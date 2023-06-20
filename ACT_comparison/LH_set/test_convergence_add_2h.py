import numpy as np
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
from getdist import plots, MCSamples
import sys
sys.path.append('/home/cemoser/Projection_Codes/Mop-c-GT-copy/mopc')
import test_mcmc_functions_add_2h as mf

suite=sys.argv[1]
prof=sys.argv[2]
z=0.54
nu=150.
#These params will go into the emulator
omega_m_set = 0.3
sigma8_set = 0.8
z_set = 0.54
mass_set = 12.25

identity='test_add_2h'

home_plots = '/home/cemoser/Repositories/emu_CAMELS/ACT_comparison/LH_set/output/'
home_chains='/home/cemoser/Repositories/emu_CAMELS/ACT_comparison/LH_set/chains/'
home_ACT_data='/home/cemoser/Repositories/emu_CAMELS/ACT_comparison/LH_set/'
home_mop='/home/cemoser/Repositories/emu_CAMELS/mopc_profiles/'
#-------------------------------------------------
sr2sqarcmin = 3282.8 * 60.**2
thta_act,proj_data,cov,cov_diag=mf.load_ACT_data(home_ACT_data,prof)
#these are now in muK*sr

ndim=5
labels = ["ASN1","AAGN1", "ASN2","AAGN2","A2h"]
labels_getdist=[r'A_{SN1}',r'A_{AGN1}',r'A_{SN2}',r'A_{AGN2}',r'A_{2h}']

number_runs=3
nwalkers=300
cut=300
itr=1000
print("convergence for %s %s with %i nwalkers, %i iter, cut %i"%(suite,prof,nwalkers,itr,cut))
print("initial points: [1.0,1.0,1.0,1.0,1.0], %s"%identity)
emu_samples,x,y,emulator=mf.build_emulator(home_mop,suite,prof)
#-----------------------------------------------------

chains_all,chains_all_reshaped,chains_all_reshaped_cut=mf.load_chains(home_chains,suite,prof,identity,nwalkers,itr,ndim,cut)
np.savetxt(home_chains+'all_chains_'+suite+'_'+prof+'_'+identity+'.txt',chains_all_reshaped_cut)

samples=chains_all_reshaped
test=samples.reshape((number_runs*nwalkers,itr,ndim))
res=test[:,cut:,:]
res_reshaped=res.reshape((-1,ndim))

c=ChainConsumer()
c.add_chain(res.reshape((-1,ndim)),walkers=number_runs*nwalkers)
gr_converged=c.diagnostic.gelman_rubin(threshold=0.1)
print(gr_converged)

#logprob to find the best sample
logprob_all_reshaped=mf.load_logprob(home_chains,suite,prof,identity,cut)
logprob_max_index=list(logprob_all_reshaped).index(np.max(logprob_all_reshaped))
logprob_sample_best=chains_all_reshaped_cut[logprob_max_index,:]


#corner plot
names=["x%s"%i for i in range(ndim)]
samp=MCSamples(samples=chains_all_reshaped_cut,names=names,labels=labels_getdist)
g=plots.getSubplotPlotter()
g.settings.figure_legend_frame=False
g.settings.title_limit_fontsize=14
g.settings.title_limit_labels=False
g.triangle_plot([samp],filled=True,title_limit=1)
plt.savefig(home_plots+'corner_'+suite+'_'+prof+'_'+identity+'.png',bbox_inches='tight')

asn1_mcmc,agn1_mcmc,asn2_mcmc,agn2_mcmc,a2h_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(chains_all_reshaped_cut, [16, 50, 84],axis=0)))
theta_fit=[asn1_mcmc[0],agn1_mcmc[0],asn2_mcmc[0],agn2_mcmc[0],a2h_mcmc[0]]

for i in range(ndim):
    mcmc = np.percentile(res_reshaped[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.2f}_{{-{1:.2f}}}^{{+{2:.2f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    latex = "${0:.2f}_{{-{1:.2f}}}^{{+{2:.2f}}}$"
    latex = latex.format(mcmc[1], q[0], q[1], labels[i])
    print(latex)

#fits
fits=[]
subsample_indices=np.random.choice(len(chains_all_reshaped_cut),size=50,replace=False)
for i in subsample_indices:
    names=chains_all_reshaped_cut[i]
    theta=[[omega_m_set,sigma8_set,names[0],names[1],names[2],names[3],z_set,mass_set]]

    fit=emulator(theta).reshape(len(x))
    fit=10**fit #undo the log
    fit=mf.project_profiles(prof,thta_act,z,nu,mf.fBeamF_150,mf.respT_150,x,fit,names[4])
    fit*=mf.rescaling_factor_dict[prof]
    if prof == 'pth_mean':
        fit = mf.add_dust(fit)
    fits.append(sr2sqarcmin*fit)

fit_84=np.percentile(fits,84,axis=0)
fit_16=np.percentile(fits,16,axis=0)
fit_50=np.percentile(fits,50,axis=0)

emu_sample_best=np.array([omega_m_set,sigma8_set,logprob_sample_best[0],logprob_sample_best[1],logprob_sample_best[2],logprob_sample_best[3],z_set,mass_set])
print("logprob best sample",logprob_sample_best)
fit_best=emulator([emu_sample_best]).reshape(len(x))
fit_best=10**fit_best #undo the log
fit_best=mf.project_profiles(prof,thta_act,z,nu,mf.fBeamF_150,mf.respT_150,x,fit_best,logprob_sample_best[4])
fit_best*=mf.rescaling_factor_dict[prof]
if prof == 'pth_mean':
    fit_best = mf.add_dust(fit_best)


min_chi2,pte=mf.chi2_pte(np.array(proj_data),np.array(fit_best),np.array(cov))
print("chi2 of the best sample found by max loprob",min_chi2)
print("pte", pte)

np.savetxt(home_plots+'fit_file_'+suite+'_'+prof+'_'+identity+'.txt',np.c_[thta_act,fit_best*sr2sqarcmin,fit_50,fit_84,fit_16],header='theta(arcmin),best SZ fit(muK*sqarcmin,max logprob),median fit, 84th percentile,16th percentile \n best fit params: %.2f,%.2f,%.2f,%.2f, A2h %.2f \n chi2 %.2f with PTE %.2f'%(logprob_sample_best[0],logprob_sample_best[1],logprob_sample_best[2],logprob_sample_best[3],logprob_sample_best[4],min_chi2,pte))


plt.figure()
if prof == 'rho_mean':
    plt.yscale('log')
    plt.ylabel(r'$T_{kSZ} [\mu K*{arcmin}^2]$')
elif prof == 'pth_mean':
    plt.ylabel(r'$T_{tSZ} [\mu K*{arcmin}^2]$')

plt.errorbar(thta_act,proj_data*sr2sqarcmin,yerr=cov_diag*sr2sqarcmin,color='black',label='ACT 150 GHz')
plt.plot(thta_act,fit_best*sr2sqarcmin,color='r',label=r'best fit: [%.2f,%.2f,%.2f,%.2f] $A_{2h}$ = %.2f, $\chi^2=$ %.2f'%(logprob_sample_best[0],logprob_sample_best[1],logprob_sample_best[2],logprob_sample_best[3],logprob_sample_best[4],min_chi2))

plt.xlabel(r'$\theta$ (arcmin)')
plt.legend()
plt.title(suite,y=0.99)
plt.savefig(home_plots+'fit_'+suite+'_'+prof+'_'+identity+'.png',bbox_inches='tight')
plt.close()
