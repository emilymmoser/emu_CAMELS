import numpy as np
import warnings
import random
import time
import helper_functions as fs
import sys
from getdist import plots,MCSamples
import matplotlib.pyplot as plt

#NOTE: this script only works for rho_mean and pth_mean, since we are projecting weighted profiles into kSZ and tSZ (can't weight the median profiles, only weighted mean)

suite=sys.argv[1]

z=fs.choose_redshift(suite)
Z_deriv=z[-1] #hard-coded for z=0.54

vary_arr=['ASN1','AAGN1','ASN2','AAGN2']


derivatives_rho=np.genfromtxt('/home/cemoser/Repositories/emu_CAMELS/derivative_arrays/'+suite+'_rho_mean_w.txt')    
derivatives_pth=np.genfromtxt('/home/cemoser/Repositories/emu_CAMELS/derivative_arrays/'+suite+'_pth_mean_w.txt')

dmu0_rho=np.transpose(derivatives_rho)
dmu0_pth=np.transpose(derivatives_pth)

#calculate the fisher matrix using the SO covmats
cov_rho=np.loadtxt('/home/cemoser/Projection_Codes/CovMat_SO_V3/CovMatV3_CMB_mode_2_fsky_0.4_beam1.4_Lmax19998.0_2018-02-19.txt',dtype=float)
cov_pth=np.loadtxt('/home/cemoser/Projection_Codes/CovMat_SO_V3/CovMatV3_y_mode_2_fsky_0.4_beam1.4_Lmax19998.0_2018-02-19.txt',dtype=float)
NGAL=498.6e4
cov_rho /= NGAL
cov_pth /= NGAL

fisher_matrix_rho=np.dot(np.transpose(dmu0_rho),np.dot(np.linalg.inv(cov_rho),dmu0_rho))
fisher_matrix_pth=np.dot(np.transpose(dmu0_pth),np.dot(np.linalg.inv(cov_pth),dmu0_pth))
fisher_combined=fisher_matrix_rho+fisher_matrix_pth


mean=np.ones(len(vary_arr))
covariance_rho=np.linalg.inv(fisher_matrix_rho)
covariance_pth=np.linalg.inv(fisher_matrix_pth)
covariance_combined=np.linalg.inv(fisher_combined)

err_rho=np.sqrt(np.diag(covariance_rho))
err_pth=np.sqrt(np.diag(covariance_pth))
err_combined=np.sqrt(np.diag(covariance_combined))

print(suite,'weighted')
print("err rho",err_rho)
print("err pth",err_pth)
print("err combined",err_combined)

np.savetxt('/home/cemoser/Repositories/emu_CAMELS/figures/corner_plots/errs_'+suite+'_w.txt',(err_rho,err_pth,err_combined),header='first line err rho, second line err pth, third line err combined \n columns ASN1, AAGN1, ASN2, AAGN2')

chain_rho=np.random.multivariate_normal(mean,covariance_rho,size=10000)
chain_pth=np.random.multivariate_normal(mean,covariance_pth,size=10000)
chain_combined=np.random.multivariate_normal(mean,covariance_combined,size=10000)

samp_rho=MCSamples(samples=chain_rho,names=vary_arr,labels=vary_arr,label=suite+r' $\rho$') 
samp_pth=MCSamples(samples=chain_pth,names=vary_arr,labels=vary_arr,label=suite+r' $P_{th}$')
samp_combined=MCSamples(samples=chain_combined,names=vary_arr,labels=vary_arr,label=r'$\rho + P_{th}$')
samp_combined.updateSettings({'contours':[0.68]})


g=plots.getSubplotPlotter()
g.settings.figure_legend_frame=False
g.settings.title_limit_fontsize=13
g.settings.title_limit_labels=False
g.triangle_plot([samp_rho,samp_pth,samp_combined],filled=True)
plt.savefig('/home/cemoser/Repositories/emu_CAMELS/figures/corner_plots/corner_2d_'+suite+'_w.png',bbox_inches='tight')
plt.close()


g=plots.get_subplot_plotter()
g.settings.title_limit_labels=False
g.settings.title_limit_fontsize=13
g.plots_1d([samp_combined,samp_pth,samp_rho],colors=['b','r','k'],title_limit=1,legend_ncol=3)
plt.savefig('/home/cemoser/Repositories/emu_CAMELS/figures/corner_plots/corner_1d_'+suite+'_w.png',bbox_inches='tight')
plt.close()
