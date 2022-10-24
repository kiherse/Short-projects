# -*- coding: utf-8 -*-
"""
@author: Kiara Hervella Seoane
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from astropy.io import fits
from matplotlib import rc

rc('text', usetex=True)

#Importamos datos para el Sol en calma
data1=fits.getdata('sol.fits')
i_s=data1[0,:,:,:]
q_s=data1[1,:,:,:]
u_s=data1[2,:,:,:]
v_s=data1[3,:,:,:]

ldo1=np.loadtxt('ldo_s')

#Importamos datos para la mancha
data2=fits.getdata('mancha.fits')
i_m=data2[0,:,:,:]
q_m=data2[1,:,:,:]
u_m=data2[2,:,:,:]
v_m=data2[3,:,:,:]

ldo2=np.loadtxt('ldo_m')

# #___________________IDENTIFICACIÓN DE ESTRUCTURAS______________________________
###############################################################################

# MAPA DE INTENSIDAD DEL CONTINUO 

# fig0=plt.figure(figsize=(13,5))
# spec=gridspec.GridSpec(ncols=2,nrows=1)
# ax0=fig0.add_subplot(spec[0])
# plt.imshow(np.mean(i_s[:,:,37:60],2),cmap='binary_r')
# plt.xlabel("pixel")
# plt.ylabel("pixel")
# divider0=make_axes_locatable(ax0)
# cax0=divider0.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax0)
# plt.colorbar(cax=cax0).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax0).set_label(r'\textbf{Intensidad}', rotation=270)
# ax1=fig0.add_subplot(spec[1])
# plt.imshow(np.mean(i_m[:,:,37:60],2),cmap='binary_r') #vmax=1.3
# plt.xlabel("pixel")
# plt.ylabel("pixel")
# divider1=make_axes_locatable(ax1)
# cax1=divider1.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax1)
# plt.colorbar(cax=cax1).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax1).set_label(r'\textbf{Intensidad}', rotation=270)
# plt.savefig('mapa_intensidad.png')

# POLARIZACIÓN CIRCULAR
pc_s=np.sqrt(v_s**2)
pc_m=np.sqrt(v_m**2)

# POLARIZACIÓN LINEAL
pl_s=np.sqrt(q_s**2+u_s**2)
pl_m=np.sqrt(q_m**2+u_m**2)

# POLARIZACIÓN TOTAL
pt_s=np.sqrt(q_s**2+u_s**2+v_s**2)
pt_m=np.sqrt(q_m**2+u_m**2+v_m**2)

# fig1=plt.figure(figsize=(13,13))
# spec=gridspec.GridSpec(ncols=2,nrows=2)
# ax0=fig1.add_subplot(spec[0,0])
# plt.imshow(np.mean(i_s[:,:,37:60],2),cmap='binary_r')
# plt.xlabel("pixel")
# plt.ylabel("pixel")
# divider0=make_axes_locatable(ax0)
# cax0=divider0.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax0)
# plt.colorbar(cax=cax0).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax0).set_label(r'\textbf{Intensidad}', rotation=270)
# ax1=fig1.add_subplot(spec[0,1])
# plt.imshow(np.mean(pc_s,2),cmap='binary_r')
# plt.xlabel("pixel")
# plt.ylabel("pixel")
# divider1=make_axes_locatable(ax1)
# cax1=divider1.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax1)
# plt.colorbar(cax=cax1).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax1).set_label(r'\textbf{Polarización circular}', rotation=270)
# ax2=fig1.add_subplot(spec[1,0])
# plt.imshow(np.mean(pl_s,2),cmap='binary_r')
# plt.xlabel("pixel")
# plt.ylabel("pixel")
# divider2=make_axes_locatable(ax2)
# cax2=divider2.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax2)
# plt.colorbar(cax=cax2).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax2).set_label(r'\textbf{Polarización lineal}', rotation=270)
# ax3=fig1.add_subplot(spec[1,1])
# plt.imshow(np.mean(pt_s,2),cmap='binary_r')
# plt.xlabel("pixel")
# plt.ylabel("pixel")
# divider3=make_axes_locatable(ax3)
# cax3=divider3.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax3)
# plt.colorbar(cax=cax3).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax3).set_label(r'\textbf{Polarización total}', rotation=270)
# plt.savefig('pol_calma.png')

# fig2=plt.figure(figsize=(13,13))
# spec=gridspec.GridSpec(ncols=2,nrows=2)
# ax0=fig2.add_subplot(spec[0,0])
# plt.imshow(np.mean(i_m[:,:,37:60],2),cmap='binary_r')
# plt.xlabel("pixel")
# plt.ylabel("pixel")
# divider0=make_axes_locatable(ax0)
# cax0=divider0.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax0)
# plt.colorbar(cax=cax0).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax0).set_label(r'\textbf{Intensidad}', rotation=270)
# ax1=fig2.add_subplot(spec[0,1])
# plt.imshow(np.mean(pc_m,2),cmap='binary_r')
# plt.xlabel("pixel")
# plt.ylabel("pixel")
# divider1=make_axes_locatable(ax1)
# cax1=divider1.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax1)
# plt.colorbar(cax=cax1).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax1).set_label(r'\textbf{Polarización circular}', rotation=270)
# ax2=fig2.add_subplot(spec[1,0])
# plt.imshow(np.mean(pl_m,2),cmap='binary_r')
# plt.xlabel("pixel")
# plt.ylabel("pixel")
# divider2=make_axes_locatable(ax2)
# cax2=divider2.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax2)
# plt.colorbar(cax=cax2).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax2).set_label(r'\textbf{Polarización lineal}', rotation=270)
# ax3=fig2.add_subplot(spec[1,1])
# plt.imshow(np.mean(pt_m,2),cmap='binary_r')
# plt.xlabel("pixel")
# plt.ylabel("pixel")
# divider3=make_axes_locatable(ax3)
# cax3=divider3.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax3)
# plt.colorbar(cax=cax3).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax3).set_label(r'\textbf{Polarización total}', rotation=270)
# plt.savefig('pol_mancha.png')


# #ESPECTROGRAMAS

# fig3=plt.figure(figsize=(10,10))
# fig=fig3
# spec=gridspec.GridSpec(ncols=4,nrows=2)
# ax0=fig.add_subplot(spec[0,0])
# plt.imshow(i_s[:,202,:])
# plt.xlabel(r"$\lambda$")
# plt.ylabel("píxel")
# divider0=make_axes_locatable(ax0)
# cax0=divider0.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax0)
# plt.colorbar(cax=cax0).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax0).set_label(r'\textbf{I}',rotation=270)
# ax1=fig.add_subplot(spec[0,1])
# plt.imshow(q_s[:,202,:])
# plt.xlabel(r"$\lambda$")
# divider1=make_axes_locatable(ax1)
# cax1=divider1.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax1)
# plt.colorbar(cax=cax1).ax.get_yaxis().labelpad=10
# plt.colorbar(cax=cax1).set_label(r'\textbf{Q/I}',rotation=270)
# ax2=fig.add_subplot(spec[0,2])
# plt.imshow(u_s[:,202,:])
# plt.xlabel(r"$\lambda$")
# divider2=make_axes_locatable(ax2)
# cax2=divider2.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax2)
# plt.colorbar(cax=cax2).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax2).set_label(r'\textbf{U/I}',rotation=270) 
# ax3=fig.add_subplot(spec[0,3])
# plt.imshow(v_s[:,202,:])
# plt.xlabel(r"$\lambda$")
# divider3=make_axes_locatable(ax3)
# cax3=divider3.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax3)
# plt.colorbar(cax=cax3).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax3).set_label(r'\textbf{V/I}',rotation=270)
# ax4=fig.add_subplot(spec[1,0])
# plt.imshow(i_m[:,202,:])
# plt.xlabel(r"$\lambda$")
# plt.ylabel("píxel")
# divider4=make_axes_locatable(ax4)
# cax4=divider4.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax4)
# plt.colorbar(cax=cax4).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax4).set_label(r'\textbf{I}',rotation=270)
# ax5=fig.add_subplot(spec[1,1])
# plt.imshow(q_m[:,202,:])
# plt.xlabel(r"$\lambda$")
# divider5=make_axes_locatable(ax5)
# cax5=divider5.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax5)
# plt.colorbar(cax=cax5).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax5).set_label(r'\textbf{Q/I}',rotation=270)
# ax6=fig.add_subplot(spec[1,2])
# plt.imshow(u_m[:,202,:])
# plt.xlabel(r"$\lambda$")
# divider6=make_axes_locatable(ax6)
# cax6=divider6.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax6)
# plt.colorbar(cax=cax6).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax6).set_label(r'\textbf{U/I}',rotation=270)
# ax7=fig.add_subplot(spec[1,3])
# plt.imshow(v_m[:,202,:])
# plt.xlabel(r"$\lambda$")
# divider7=make_axes_locatable(ax7)
# cax7=divider7.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax7)
# plt.colorbar(cax=cax7).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax7).set_label(r'\textbf{V/I}',rotation=270)
# plt.savefig("espectrogramas.png")


#_____________________MEDIDAS DEL CAMPO MAGNÉTICO______________________________
###############################################################################

C=4.67e-13 #en Gauss y amstrongs
g_eff1=1.667 #para la primera línea
g_eff2=2.5 #para la segunda línea
l1=6301.52
l2=6302.51

# #CAMPO MAGNÉTICO DE LA MANCHA: aproximación de campo fuerte

B_m=np.zeros((399,401))
ldoB=np.zeros((399,401))

for j in range(399):
    for k in range(401):
        try:
            
            index_min=np.where(v_m[j,k,50:90]==np.min(v_m[j,k,50:90]))[0][0]
            l_min=ldo2[index_min+50-3:index_min+50+3]
            i_min=v_m[j,k,index_min+50-3:index_min+50+3]
            a_min,b_min,c_min=np.polyfit(l_min,i_min,2)
            min_min=-b_min/(2*a_min)
            
            index_max=np.where(v_m[j,k,50:90]==np.max(v_m[j,k,50:90]))[0][0]
            l_max=ldo2[index_max+50-3:index_max+50+3]
            i_max=v_m[j,k,index_max+50-3:index_max+50+3]
            a_max,b_max,c_max=np.polyfit(l_max,i_max,2)
            min_max=-b_max/(2*a_max)
            
            ldoB[j,k]=min_max-min_min            
            B_m[j,k]=np.abs((min_max-min_min)/(2*C*g_eff2*(6302.51**2)))
       
        except:
            # print("Fallo en: %d %d" %(j,k))
            continue


doppler_m=np.sqrt((8*np.log(2)*1.38e-23*T_m)/(9.27e-26*(3e8)**2))*(6302.51)
ldoB-doppler_m[:,:,5]


 
plt.figure(figsize=(10,10)) #Campo magnético estimado con 6302.51
ax=plt.gca()
im=ax.imshow(B_m,vmin=0,vmax=3200)
plt.xlabel("píxel")
plt.ylabel("píxel")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
cbar=plt.colorbar(im, cax=cax)
cbar.ax.get_yaxis().labelpad=14
cbar.set_label(r'\textbf{B [G]}', rotation=270)
plt.savefig('campo_fuerte.png')

# Buscamos un punto de máximo campo: [190,225]
# El parámetro de Stokes indica polarización negativa 
# Por lo que normalizamos a 180 grados el centro de la mancha

#Buscamos promediar el mínimo de la intensidad (colapsando en las dos direcciones)
# v_1=np.zeros((399,401))
# index_min=np.zeros((399,401))
# for j in range(399):
#     for k in range(401):
#         index=np.where(i_s[j,k,:]==np.min(i_s[j,k,50:]))[0][0]
#         l_vs=ldo1[index-4:index+4]
#         i_vs=i_s[j,k,index-4:index+4]
#         a_vs,b_vs,c_vs=np.polyfit(l_vs,i_vs,2)
#         min_vs=-b_vs/(2*a_vs)
#         v_1[j,k]=min_vs
#         index_min[j,k]=index
        
        
        
# index_max=np.zeros((399,401))
# v_2=np.zeros((399,401))
# for j in range(399):
#     for k in range(401):
#         index=np.where(i_s[j,k,:]==np.max(i_s[j,k,50:]))[0][0]
#         l_vs=ldo1[index-4:index+4]
#         i_vs=i_s[j,k,index-4:index+4]
#         a_vs,b_vs,c_vs=np.polyfit(l_vs,i_vs,2)
#         min_vs=-b_vs/(2*a_vs)
#         v_2[j,k]=min_vs
#         index_max[j,k]=index
        
# dif=index_max-index_min
# v_21=v_2-v_1
# index_v=np.where(dif[:,:]==np.max(dif[:,:]))
# # index=np.where(v_m[index_max[0],index_max[1],:]==np.max(v_m[index_max[0],index_max[1],:]))
# v_maxs=v_m[index_v[0],index_v[1],:]
# index_v_max=np.where(v_maxs[:,:]==np.max(v_maxs[:,:]))
# v_max=v_maxs[index_v_max[0],index_v_max[1]]

v_mean=np.mean(v_m[:,:,35:55],2)
v_max=np.amax(np.mean(v_m[:,:,35:55],2))
cosinc=-v_mean/0.03 #por ahora lo dejo asi para que la grafica quede chula :)
for i in range(0,399):
  for j in range(0,401):
    if cosinc[i,j]>1:
      cosinc[i,j]=1
    if cosinc[i,j]<-1:
      cosinc[i,j]=-1

gamma_m=np.arccos(cosinc)

# plt.figure(figsize=(10,10))
# ax=plt.gca()
# im=plt.imshow((180/np.pi)*(gamma_m[:,:])) #Ploteamos normalizado
# plt.xlabel("píxel")
# plt.ylabel("píxel")
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.2)
# cbar=plt.colorbar(im, cax=cax)
# cbar.ax.get_yaxis().labelpad=14
# cbar.set_label(r'\textbf{$\gamma$} [grados]', rotation=270)
# plt.savefig('gamma_mancha.png')

# fi_m=0.5*np.arctan(u_m/q_m) #azimuth

# plt.figure(figsize=(10,10))
# ax=plt.gca()
# im=plt.imshow((180/np.pi)*fi_m[:,:,40])
# plt.xlabel("píxel")
# plt.ylabel("píxel")
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.2)
# cbar=plt.colorbar(im, cax=cax)
# cbar.ax.get_yaxis().labelpad=14
# cbar.set_label(r'\textbf{$\Phi$} [grados]', rotation=270)
# plt.savefig('fi_mancha.png')

            
# # #CAMPO MAGNÉTICO DEL SOL EN CALMA: aproximación de campo débil
Ldo1=np.reshape(ldo1,96)

i_der=np.diff(i_s[:,:,50:],1)/np.diff(Ldo1[50:96])
i_der2=np.diff(i_der[:,:,:],1)/np.diff(Ldo1[50:95])

Bpar=-np.sum(v_s[:,:,50:95]*i_der,2)/(C*g_eff2*(l2**2)*np.sum(i_der*i_der,2))
Bper=np.sqrt((4*np.sqrt((np.sum(q_s[:,:,50:94]*i_der2,2))**2+(np.sum(u_s[:,:,50:94]*i_der2,2))**2))/((C**2)*(l2**4)*(g_eff2**2)*np.sum(i_der2*i_der2,2)))
Btotal=np.sqrt(Bpar**2+Bper**2)

fi0=np.zeros((401,401))

for i in range(401):
    for j in range(401):
        
        if np.sum(q_s[i,j,50:94]*i_der2[i,j])>=0:
            fi0[i,j]=np.pi/2            
            
        else:
            if np.sum(u_s[i,j,50:94]*i_der2[i,j])<=0:
                fi0[i,j]=0
                
            else:
                fi0[i,j]=np.pi               
            
fi_s=0.5*np.arctan(np.sum(u_s[:,:,50:94]*i_der2,2)/np.sum(u_s[:,:,50:94]*i_der2,2))+fi0

plt.figure(figsize=(10,10))
ax=plt.gca()
im=plt.imshow(Btotal)
plt.xlabel("píxel")
plt.ylabel("píxel")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
cbar=plt.colorbar(im, cax=cax)
cbar.ax.get_yaxis().labelpad=14
cbar.set_label(r'\textbf{B [G]}', rotation=270)
plt.savefig('campo_debil.png')

plt.figure(figsize=(10,10))
ax=plt.gca()
im=plt.imshow(Bpar)
plt.xlabel("píxel")
plt.ylabel("píxel")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
cbar=plt.colorbar(im, cax=cax)
cbar.ax.get_yaxis().labelpad=14
cbar.set_label(r'\textbf{$B_{par}$ [G]}', rotation=270)
plt.savefig('campo_debil_paralelo.png')

plt.figure(figsize=(10,10))
ax=plt.gca()
im=plt.imshow(Bper)
plt.xlabel("píxel")
plt.ylabel("píxel")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
cbar=plt.colorbar(im, cax=cax)
cbar.ax.get_yaxis().labelpad=14
cbar.set_label(r'\textbf{$B_{per}$ [G]}', rotation=270)
plt.savefig('campo_debil_perpendicular.png')


plt.figure(figsize=(10,10))
ax=plt.gca()
im=plt.imshow((180/np.pi)*fi_s)
plt.xlabel("píxel")
plt.ylabel("píxel")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
cbar=plt.colorbar(im, cax=cax)
cbar.ax.get_yaxis().labelpad=14
cbar.set_label(r'\textbf{$\Phi$ [grados]}', rotation=270)
plt.savefig('fi_debil.png')

# ldoBs=np.zeros((401,401))

# for j in range(401):
#     for k in range(401):
#         try:
            
#             index_min=np.where(v_s[j,k,50:90]==np.min(v_s[j,k,50:90]))[0][0]
#             l_min=ldo1[index_min+50-3:index_min+50+3]
#             i_min=v_m[j,k,index_min+50-3:index_min+50+3]
#             a_min,b_min,c_min=np.polyfit(l_min,i_min,2)
#             min_min=-b_min/(2*a_min)
            
#             index_max=np.where(v_s[j,k,50:90]==np.max(v_s[j,k,50:90]))[0][0]
#             l_max=ldo1[index_max+50-3:index_max+50+3]
#             i_max=v_m[j,k,index_max+50-3:index_max+50+3]
#             a_max,b_max,c_max=np.polyfit(l_max,i_max,2)
#             min_max=-b_max/(2*a_max)
            
#             ldoBs[j,k]=min_max-min_min            
       
#         except:
#             # print("Fallo en: %d %d" %(j,k))
#             continue

# doppler_s=np.sqrt((8*np.log(2)*1.38e-23*T_s)/(9.27e-26*(3e8)**2))*(6302.51)

# fig0=plt.figure(figsize=(13,5))
# spec=gridspec.GridSpec(ncols=2,nrows=1)
# ax0=fig0.add_subplot(spec[0])
# plt.imshow(ldoB-doppler_m[:,:,5],vmin=-0.2,vmax=0.2)
# plt.xlabel("píxel")
# plt.ylabel("píxel")
# divider0=make_axes_locatable(ax0)
# cax0=divider0.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax0)
# plt.colorbar(cax=cax0).ax.get_yaxis().labelpad=14
# plt.colorbar(cax=cax0).set_label(r'\textbf{$\Delta\lambda_B-\Delta\lambda_D$} [$\AA$]',rotation=270)
# ax1=fig0.add_subplot(spec[1])
# plt.imshow(doppler_s[:,:,5]-ldoBs,vmin=-1,vmax=1)
# plt.xlabel("píxel")
# plt.ylabel("píxel")
# divider1=make_axes_locatable(ax1)
# cax1=divider1.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax1)
# plt.colorbar(cax=cax1).ax.get_yaxis().labelpad=14
# plt.colorbar(cax=cax1).set_label(r'\textbf{$\Delta\lambda_D-\Delta\lambda_B$} [$\AA$]',rotation=270)
# plt.savefig('validez.png')


#___________________MEDIDAS DE TEMPERATURA Y VELOCIDAD_________________________
###############################################################################

# #PERFILES DE UN GRÁNULO Y UN INTERGRÁNULO

# #Para buscar el intergránulo y el gránulo acudimos al mapa de intensidad
# plt.figure(19)
# plt.title('Mapa de intensidad del continuo para el Sol en calma')
# plt.imshow(np.mean(i_s[:,:,37:60],2))
# plt.colorbar()
# plt.show()

# plt.figure(11)
# plt.figure(figsize=(10,10))
# plt.title('Perfiles de un gránulo y un intergránulo del Sol en calma')
# plt.xlabel(r'$\lambda$ [$\AA$]')
# plt.ylabel('I')
# plt.plot(ldo1,i_s[98,169,:],label='Gránulo (356,127)')
# plt.plot(ldo1,i_s[101,182,:],label='Intergranulo (351,122)')
# plt.plot(ldo1,np.mean(np.mean(i_s,0),0),label='Promedio')
# plt.grid()
# plt.legend()
# plt.savefig('granulo_intergranulo.png')

#MAPAS DE TEMPERATURA
h=6.63e-34
k=1.38e-23
c=3e8
Teff=5780

T_s=1/(1/Teff-(k*np.log(i_s[:,:,37:60])*ldo1[37:60]*1e-10)/(h*c))
T_m=1/(1/Teff-(k*np.log(i_m[:,:,37:60])*ldo1[37:60]*1e-10)/(h*c))

# fig0=plt.figure(figsize=(13,5))
# spec=gridspec.GridSpec(ncols=2,nrows=1)
# ax0=fig0.add_subplot(spec[0])
# plt.imshow(T_s[:,:,5])
# plt.xlabel("píxel")
# plt.ylabel("píxel")
# divider0=make_axes_locatable(ax0)
# cax0=divider0.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax0)
# plt.colorbar(cax=cax0).ax.get_yaxis().labelpad=14
# plt.colorbar(cax=cax0).set_label(r'\textbf{T [K]}',rotation=270)
# ax1=fig0.add_subplot(spec[1])
# plt.imshow(T_m[:,:,5])
# plt.xlabel("píxel")
# plt.ylabel("píxel")
# divider1=make_axes_locatable(ax1)
# cax1=divider1.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax1)
# plt.colorbar(cax=cax1).ax.get_yaxis().labelpad=14
# plt.colorbar(cax=cax1).set_label(r'\textbf{T [K]}',rotation=270)
# plt.savefig('temperatura.png')

#MAPAS DE VELOCIDAD
# #calculamos las velocidades mediante efecto Doppler
# #debemos restar la velocidad promedio de una región en calma

# #Buscamos promediar el mínimo de la intensidad (colapsando en las dos direcciones)
# vel_s=np.zeros((401,401))
# for j in range(401):
#     for k in range(401):
#         index=np.where(i_s[j,k,:]==np.min(i_s[j,k,:50]))[0][0]
#         l_vs=ldo1[index-4:index+4]
#         i_vs=i_s[j,k,index-4:index+4]
#         a_vs,b_vs,c_vs=np.polyfit(l_vs,i_vs,2)
#         min_vs=-b_vs/(2*a_vs)
#         vel_s[j,k]=(c*(min_vs-6301.52))/6301.52
        
# vel_m=np.zeros((399,401))
# for j in range(399):
#     for k in range(401):
#         index=np.where(i_m[j,k,:]==np.min(i_m[j,k,:50]))[0][0]
#         l_vm=ldo2[index-8:index+8]
#         i_vm=i_m[j,k,index-8:index+8]
#         a_vm,b_vm,c_vm=np.polyfit(l_vm,i_vm,2)
#         min_vm=-b_vm/(2*a_vm)
#         vel_m[j,k]=(c*(min_vm-6301.52))/6301.52
        
# #Tenemos problemas con las velocidades en la mancha

# fig0=plt.figure(figsize=(13,5))
# spec=gridspec.GridSpec(ncols=2,nrows=1)
# ax0=fig0.add_subplot(spec[0])
# plt.imshow(vel_s,cmap="seismic")
# plt.xlabel("píxel")
# plt.ylabel("píxel")
# divider0=make_axes_locatable(ax0)
# cax0=divider0.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax0)
# plt.colorbar(cax=cax0).ax.get_yaxis().labelpad=14
# plt.colorbar(cax=cax0).set_label(r'\textbf{v [m/s]}',rotation=270)
# ax1=fig0.add_subplot(spec[1])
# plt.imshow(vel_m,cmap="seismic",vmin=-2000,vmax=2000)
# plt.xlabel("píxel")
# plt.ylabel("píxel")
# divider1=make_axes_locatable(ax1)
# cax1=divider1.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax1)
# plt.colorbar(cax=cax1).ax.get_yaxis().labelpad=14
# plt.colorbar(cax=cax1).set_label(r'\textbf{v [m/s]}',rotation=270)
# plt.savefig('velocidad.png')
      
                 
# TEMPERATURA FRENTE A VELOCIDAD
# representamos la temperatura de cada pixel frente a su velocidad

# fig0=plt.figure(figsize=(13,5))
# spec=gridspec.GridSpec(ncols=2,nrows=1)
# ax0=fig0.add_subplot(spec[0])
# plt.plot(T_s[:,:,12],vel_s,'.')
# plt.xlabel("T [K]")
# plt.ylabel("v [m/s]")
# ax1=fig0.add_subplot(spec[1])
# plt.plot(T_m[:,:,12],vel_m,'.')
# plt.xlabel("T [K]")
# plt.ylabel("v [m/s]")
# plt.savefig('velocidad_temperatura.png')
