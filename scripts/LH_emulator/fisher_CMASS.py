import numpy as np
import helper_functions_LH as fs
import sys
from getdist import plots,MCSamples
import matplotlib.pyplot as plt

suite=sys.argv[1]
project_type=sys.argv[2] #either fft or gauss

z=fs.choose_redshift(suite)
Z_deriv=z[-1] #hard-coded for z=0.54 (for CMASS)

vary_arr=['ASN1','AAGN1','ASN2','AAGN2']

#where the figures will save
save='./'

#point to derivative arrays
home_deriv='/home/cemoser/Repositories/emu_CAMELS/derivative_arrays/'
derivatives_rho=np.genfromtxt(home_deriv+suite+'_rho_mean_'+project_type+'_CMASS.txt')    
derivatives_pth=np.genfromtxt(home_deriv+suite+'_pth_mean_'+project_type+'_CMASS.txt')

dmu0_rho=np.transpose(derivatives_rho)
dmu0_pth=np.transpose(derivatives_pth)

#SO covmat, DESI-like NGAL
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


chain_rho=np.random.multivariate_normal(mean,covariance_rho,size=10000)
chain_pth=np.random.multivariate_normal(mean,covariance_pth,size=10000)
chain_combined=np.random.multivariate_normal(mean,covariance_combined,size=10000)

samp_rho=MCSamples(samples=chain_rho,names=vary_arr,labels=vary_arr,label=suite+r' kSZ') 
samp_pth=MCSamples(samples=chain_pth,names=vary_arr,labels=vary_arr,label=suite+r' tSZ')
samp_combined=MCSamples(samples=chain_combined,names=vary_arr,labels=vary_arr,label=r'tSZ + kSZ')
samp_combined.updateSettings({'contours':[0.68]})


g=plots.getSubplotPlotter()
g.settings.figure_legend_frame=False
g.settings.title_limit_fontsize=13
g.settings.title_limit_labels=False
g.triangle_plot([samp_combined,samp_rho,samp_pth],filled=True,title_limit=1,contour_args=[{'zorder':3,'color':'b'},{'zorder':1,'color':'dimgray'},{'zorder':2,'color':'r'}],label_order=[1,2,0])
plt.savefig(save+'corner_2d_'+project_type+'_'+suite+'_CMASS.png',bbox_inches='tight')
plt.close()

#1D distribution plots
'''
g=plots.get_subplot_plotter()
g.settings.title_limit_labels=False
g.settings.title_limit_fontsize=13
g.plots_1d([samp_combined,samp_pth,samp_rho],colors=['b','r','k'],title_limit=1)
plt.savefig(save+'corner_1d_'+project_type+'_'+suite+'_w.png',bbox_inches='tight')
plt.close()
'''
