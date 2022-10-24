# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 16:16:43 2021

@author: Kiara Hervella Seoane
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from astropy.io import fits
from matplotlib import rc
rc('text', usetex=True)

#_______________CORRECCIÓN DEL CONTINUO Y CALIBRACIÓN DE LONGITUD______________
###############################################################################

#Importamos datos para el Sol en calma
data1=fits.getdata('Stokes_quietsun_HINODE.fits')
np.shape(data1) #Dimensión (4, 401, 401, 96)

#Buscamos promediar el mínimo de la intensidad (colapsando en las dos direcciones)
I1=data1[0,:,:,:]
promedio1=np.mean(np.mean(I1,0),0)
#Buscamos aquí las regiones del continuo (0:13,35:62,78:) para ajustarlas a una función
y1=np.concatenate((promedio1[:13],promedio1[35:62],promedio1[78:]))
#Creamos las variables en x
X=np.linspace(1,96,96)
x=np.concatenate((X[:13],X[35:62],X[78:]))
#Ajustamos los datos
ax1=np.polyfit(x,y1,6)
#Creamos el continuo con los datos ajustados
cont1=np.polyval(ax1,X)

# plt.figure(1)
# plt.grid()
# plt.plot(I1[200,201,:])
# plt.xlabel(r"$\lambda$")
# plt.ylabel("I [cuentas]")
# plt.savefig("intensidad_ej_calma.png")


#Ahora normalizamos todos los datos de intensidad dividiendo por el continuo promediado y normalizado
i1=I1/cont1
q1=data1[1,:,:,:]/i1/cont1
u1=data1[2,:,:,:]/i1/cont1
v1=data1[3,:,:,:]/i1/cont1
  
# Podemos observar la normalización del continuo graficamente
# plt.figure(1)
# plt.grid()
# plt.plot(cont1, label="Intensidad de continuo corregida")
# plt.plot(promedio1, label="Intensidad promedio") 
# plt.xlabel(r"$\lambda$")
# plt.ylabel("I [cuentas]")
# plt.legend()
# plt.savefig("norm_cont_calma.png")



###############################################################################

#Importamos ahora nuestros datos para la mancha solar
data2=fits.getdata('Stokes_sunspot_HINODE.fits')
np.shape(data2) #Dimensión (4, 401, 401, 96)

#Para poder promediar en los datos de la mancha tenemos que tener en cuenta sólo el continuo
#Para ello graficamos un mapa de intensidad para localizar la región del sol en calma
# plt.show()
# plt.imshow(data2[0,:,:,50],cmap='binary_r')
# plt.colorbar()
#Con esto obtenemos las treinta primeras columnas (dirección y)

#Buscamos promediar el mínimo de la intensidad (colapsando en las dos direcciones)
I2=data2[0,:,:,:]
promedio2=np.mean(np.mean(I2,0),0)
y2=np.concatenate((promedio2[:13],promedio2[35:62],promedio2[78:]))
ax2=np.polyfit(x,y2,6)
cont2=np.polyval(ax2,X)

i2=I2/cont2
q2=data2[1,:,:,:]/i2/cont2
u2=data2[2,:,:,:]/i2/cont2
v2=data2[3,:,:,:]/i2/cont2

# plt.figure(2)
# plt.title('Mancha solar normalizado')
# plt.grid()
# plt.plot(cont2)
# plt.plot(promedio2) 

# plt.figure(2)
# plt.grid()
# plt.plot(cont2, label="Intensidad de continuo corregida")
# plt.plot(promedio2, label="Intensidad promedio") 
# plt.xlabel(r"$\lambda$")
# plt.ylabel("I [cuentas]")
# plt.legend()
# plt.savefig("norm_cont_mancha.png")


#_______________________CALIBRACIÓN EN LONGITUD DE ONDA________________________
###############################################################################

#Importamos el atlas
atlas=fits.getdata('atlas_6301_6302.fits') #importamos datos de una zona en calma para estudiar la granulación
# np.shape(atlas) #Tenemos las dimensiones (2000,2)
ldo=atlas[:,0]
I=atlas[:,1]

#Buscamos los mínimos visualmente
#atlas: 6301,5106 y 6302,5021
#sol y mancha solar: 24 y 70

#Tengamos en cuenta que tanto para la mancha como para el Sol en calma los minimos se encuentran en los mismos índices
#Nos llega con una ldo para los dos fragmentos de ambos: 
l1=[];l2=[]
int1_sol=[];int2_sol=[]
int1_mancha=[];int2_mancha=[]

for j in range(21,28,1):
    l1.append(X[j])
    int1_sol.append(promedio1[j])
    int1_mancha.append(promedio2[j])
            
for j in range(67,74,1):
    l2.append(X[j])
    int2_sol.append(promedio1[j])
    int2_mancha.append(promedio2[j])
    
a1_sol,b1_sol,c1_sol=np.polyfit(l1,int1_sol,2)
a2_sol,b2_sol,c2_sol=np.polyfit(l2,int2_sol,2)
a1_mancha,b1_mancha,c1_mancha=np.polyfit(l1,int1_mancha,2)
a2_mancha,b2_mancha,c2_mancha=np.polyfit(l2,int2_mancha,2)

min1_sol=-b1_sol/(2*a1_sol)
min2_sol=-b2_sol/(2*a2_sol)

min1_mancha=-b1_mancha/(2*a1_mancha)
min2_mancha=-b2_mancha/(2*a2_mancha)
      
#Buscamos ahora los mínimos en el atlas
L1=[];L2=[]
INT1=[];INT2=[]

for j in range(len(ldo)):
    if ldo[j]==6301.50:
        n1=j
    elif ldo[j]==6302.5:
        n2=j

for j in range(n1-10,n1+21,1):
    L1.append(ldo[j])
    INT1.append(I[j])

for j in range(n2-10,n2+11,1):
    L2.append(ldo[j])
    INT2.append(I[j])

a1_atlas,b1_atlas,c1_atlas=np.polyfit(L1,INT1,2)
a2_atlas,b2_atlas,c2_atlas=np.polyfit(L2,INT2,2)

min1=-b1_atlas/(2*a1_atlas)
min2=-b2_atlas/(2*a2_atlas)

#Calibramos ahora para el Sol en calma y para la mancha
min_sol=np.zeros(2);min_mancha=np.zeros(2)
min_atlas=np.zeros(2)

min_sol[0]=min1_sol
min_sol[1]=min2_sol

min_mancha[0]=min1_mancha
min_mancha[1]=min2_mancha

min_atlas[0]=min1
min_atlas[1]=min2

#Calibramos ahora para el Sol en calma
coef_sol=np.polyfit(min_sol,min_atlas,1)
ldo1=np.polyval(coef_sol,X)
 
#Calibramos ahora para la mancha
coef_mancha=np.polyfit(min_mancha,min_atlas,1)
ldo2=np.polyval(coef_mancha,X)

##############################################################################

#Pintamos los perfiles de intensidad promedio frente a estas longitudes de onda calibradas
#frente al atlas, tanto para el Sol en calma como para los datos de la mancha

#Debemos dividir la intensidad del atlas entre 10000 en un intento vago de normalización
# plt.figure(1)
# # plt.title('Sol en calma, normalizado y calibrado')
# plt.grid()
# plt.plot(ldo,I/10000,label='Intensidad del Atlas')
# plt.plot(ldo1,promedio1/cont1,label='Intensidad promedio') 
# plt.xlabel(r'$\lambda$ [nm]')
# plt.ylabel('Intensidad')
# plt.legend()
# plt.savefig('atlas_calma.png')

# plt.figure(2)
# plt.grid()
# plt.plot(ldo,I/10000,label='Intensidad del Atlas')
# plt.plot(ldo2,promedio2/cont2, label='Intensidad promedio') 
# plt.xlabel(r'$\lambda$ [nm]')
# plt.ylabel('Intensidad')
# plt.legend()
# plt.savefig('atlas_mancha.png')

# #PERFILES DE TODOS LOS PARÁMETROS
# fig1=plt.figure(figsize=(15,15))
# spec=gridspec.GridSpec(ncols=2,nrows=2)
# ax0=fig1.add_subplot(spec[0,0])
# plt.plot(ldo1,i1[200,201,:])
# plt.xlabel(r"$\lambda$ [$\AA$]")
# plt.ylabel("I")
# ax1=fig1.add_subplot(spec[1,0])
# plt.plot(ldo1,q1[200,201,:])
# plt.xlabel(r"$\lambda$ [$\AA$]")
# plt.ylabel("Q/I")
# ax2=fig1.add_subplot(spec[0,1])
# plt.plot(ldo1,u1[200,201,:])
# plt.xlabel(r"$\lambda$ [$\AA$]")
# plt.ylabel("U/I")
# ax3=fig1.add_subplot(spec[1,1])
# plt.plot(ldo1,v1[200,201,:])
# plt.xlabel(r"$\lambda$ [$\AA$]")
# plt.ylabel("V/I")
# plt.savefig("iquv_calma")

###############################################################################

#Tenemos entonces todos nuestros datos normalizados y calibrado en 
#i1,q1,u1,v1 y ldo1 para el sol en calma
#i2,q2,u2,v2 y ldo2 para los datos de la mancha solar
#I y ldo para el atlas

#___________________PCA: REDUCCIÓN DE ERRORES__________________________________
###############################################################################

#SOL EN CALMA

#Construímos na matriz D(nxp) de nuestros datos donde n=intensidad y p=ldo
#Para ello tenemos que redimensionar nuestro cubo de datos a n=401*401 y l=96
Di1=np.reshape(i1,(160801,96)) #pasa los primeros 401 datos de i1[0,0,:] y luego los 401 de i1[1,0,:]
 
#Construímos la matriz de correlación C=D^tD y la diagonalizamos
Ci1=np.dot(Di1.T,Di1) #Di1.T nos calcula la traspuesta

#Ahora diagonalizamos esta matriz
i1_val,i1_vec=np.linalg.eig(Ci1)
idx_i1=np.argsort(i1_val)
index_i1=idx_i1[::-1]
i1_Val=i1_val[index_i1]
Bi1=i1_vec[:,index_i1] #los autovectores están ordenados en columnas

#Transformamos los datos a esta base
Ai1=np.dot(Di1,Bi1)

fi1,ci1=np.shape(Ai1)
i1_new=np.dot(Ai1[:,:11],Bi1[:,:11].T)
        
#Repetimos el proceso para Q
Dq1=np.reshape(q1,(160801,96)) 
Cq1=np.dot(Dq1.T,Dq1)

q1_val,q1_vec=np.linalg.eig(Cq1)
idx_q1=np.argsort(q1_val)
index_q1=idx_q1[::-1]
q1_Val=q1_val[index_q1]
Bq1=q1_vec[:,index_q1]

Aq1=np.dot(Dq1,Bq1)

q1_new=np.dot(Aq1[:,:11],Bq1[:,:11].T)

#Repetimos el proceso para U
Du1=np.reshape(u1,(160801,96)) 
Cu1=np.dot(Du1.T,Du1)

u1_val,u1_vec=np.linalg.eig(Cu1)
idx_u1=np.argsort(u1_val)
index_u1=idx_u1[::-1]
u1_Val=u1_val[index_u1]
Bu1=u1_vec[:,index_u1]

Au1=np.dot(Du1,Bu1)

u1_new=np.dot(Au1[:,:11],Bu1[:,:11].T)
        
#Repetimos el proceso para V
Dv1=np.reshape(v1,(160801,96)) 
Cv1=np.dot(Dv1.T,Dv1)

v1_val,v1_vec=np.linalg.eig(Cv1)
idx_v1=np.argsort(v1_val)
index_v1=idx_v1[::-1]
v1_Val=v1_val[index_v1]
Bv1=v1_vec[:,index_v1]

Av1=np.dot(Dv1,Bv1)

v1_new=np.dot(Av1[:,:11],Bv1[:,:11].T)
        
###############################################################################

# MANCHA SOLAR

#Proceso para I
Di2=np.reshape(i2,(159999,96))
Ci2=np.dot(Di2.T,Di2)

i2_val,i2_vec=np.linalg.eig(Ci2)
idx_i2=np.argsort(i2_val)
index_i2=idx_i2[::-1]
i2_Val=i2_val[index_i2]
Bi2=i2_vec[:,index_i2] 

Ai2=np.dot(Di2,Bi2)

i2_new=np.dot(Ai2[:,:17],Bi2[:,:17].T)
               
#Repetimos el proceso para Q
Dq2=np.reshape(q2,(159999,96)) 
Cq2=np.dot(Dq2.T,Dq2)

q2_val,q2_vec=np.linalg.eig(Cq2)
idx_q2=np.argsort(q2_val)
index_q2=idx_q2[::-1]
q2_Val=q2_val[index_q2]
Bq2=q2_vec[:,index_q2]

Aq2=np.dot(Dq2,Bq2)

q2_new=np.dot(Aq2[:,:17],Bq2[:,:17].T)

#Repetimos el proceso para U
Du2=np.reshape(u2,(159999,96)) 
Cu2=np.dot(Du2.T,Du2)

u2_val,u2_vec=np.linalg.eig(Cu2)
idx_u2=np.argsort(u2_val)
index_u2=idx_u2[::-1]
u2_Val=u2_val[index_u2]
Bu2=u2_vec[:,index_u2]

Au2=np.dot(Du2,Bu2)

u2_new=np.dot(Au2[:,:17],Bu2[:,:17].T)
        
#Repetimos el proceso para V
Dv2=np.reshape(v2,(159999,96)) 
Cv2=np.dot(Dv2.T,Dv2)

v2_val,v2_vec=np.linalg.eig(Cv2)
idx_v2=np.argsort(v2_val)
index_v2=idx_v2[::-1]
v2_Val=v2_val[index_v2]
Bv2=v2_vec[:,index_v2]

Av2=np.dot(Dv2,Bv2)

v2_new=np.dot(Av2[:,:17],Bv2[:,:17].T)

# MODIFICAMOS LA FORMA DE LOS DATOS PARA TRABAJAR CON ELLOS 
i_s=np.reshape(i1_new,(401,401,96))
q_s=np.reshape(q1_new,(401,401,96))
u_s=np.reshape(u1_new,(401,401,96))
v_s=np.reshape(v1_new,(401,401,96))

i_m=np.reshape(i2_new,(399,401,96))
q_m=np.reshape(q2_new,(399,401,96))
u_m=np.reshape(u2_new,(399,401,96))
v_m=np.reshape(v2_new,(399,401,96))

###############################################################################

#PERFILES DE TODOS LOS PARÁMETROS PARA EL EFECTO DE PCA

# fig1=plt.figure(figsize=(15,15))
# spec=gridspec.GridSpec(ncols=2,nrows=2)
# ax0=fig1.add_subplot(spec[0,0])
# plt.plot(ldo1,i1[200,201,:])
# plt.plot(ldo1,i_s[200,201,:])
# plt.xlabel(r"$\lambda$ [$\AA$]")
# plt.ylabel("I")
# ax1=fig1.add_subplot(spec[1,0])
# plt.plot(ldo1,q1[200,201,:])
# plt.plot(ldo1,q_s[200,201,:])
# plt.xlabel(r"$\lambda$ [$\AA$]")
# plt.ylabel("Q/I")
# ax2=fig1.add_subplot(spec[0,1])
# plt.plot(ldo1,u1[200,201,:])
# plt.plot(ldo1,u_s[200,201,:])
# plt.xlabel(r"$\lambda$ [$\AA$]")
# plt.ylabel("U/I")
# ax3=fig1.add_subplot(spec[1,1])
# plt.plot(ldo1,v1[200,201,:])
# plt.plot(ldo1,v_s[200,201,:])
# plt.xlabel(r"$\lambda$ [$\AA$]")
# plt.ylabel("V/I")
# plt.savefig("pca_iquv_calma.png")

# fig1=plt.figure(figsize=(15,15))
# spec=gridspec.GridSpec(ncols=2,nrows=2)
# ax0=fig1.add_subplot(spec[0,0])
# plt.plot(ldo2,i2[200,201,:])
# plt.plot(ldo2,i_m[200,201,:])
# plt.xlabel(r"$\lambda$ [$\AA$]")
# plt.ylabel("I")
# ax1=fig1.add_subplot(spec[1,0])
# plt.plot(ldo2,q2[200,201,:])
# plt.plot(ldo2,q_m[200,201,:])
# plt.xlabel(r"$\lambda$ [$\AA$]")
# plt.ylabel("Q/I")
# ax2=fig1.add_subplot(spec[0,1])
# plt.plot(ldo2,u2[200,201,:])
# plt.plot(ldo2,u_m[200,201,:])
# plt.xlabel(r"$\lambda$ [$\AA$]")
# plt.ylabel("U/I")
# ax3=fig1.add_subplot(spec[1,1])
# plt.plot(ldo2,v2[200,201,:])
# plt.plot(ldo2,v_m[200,201,:])
# plt.xlabel(r"$\lambda$ [$\AA$]")
# plt.ylabel("V/I")
# plt.savefig("pca_iquv_mancha.png")

#  #Mapas para ver el efecto de PCA

# fig1=plt.figure(figsize=(15,15))
# spec=gridspec.GridSpec(ncols=2,nrows=4)
# ax0=fig1.add_subplot(spec[0,0])
# plt.imshow(i_s[:,:,50],cmap='binary_r')
# plt.xlabel("pixel")
# plt.ylabel("pixel")
# divider0=make_axes_locatable(ax0)
# cax0=divider0.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax0)
# plt.colorbar(cax=cax0).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax0).set_label(r'\textbf{I}', rotation=270)
# ax1=fig1.add_subplot(spec[1,0])
# plt.imshow(q_s[:,:,50],cmap='binary_r')
# plt.xlabel("pixel")
# plt.ylabel("pixel")
# divider1=make_axes_locatable(ax1)
# cax1=divider1.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax1)
# plt.colorbar(cax=cax1).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax1).set_label(r'\textbf{Q/I}', rotation=270)
# ax2=fig1.add_subplot(spec[2,0])
# plt.imshow(u_s[:,:,50],cmap='binary_r')
# plt.xlabel("pixel")
# plt.ylabel("pixel")
# divider2=make_axes_locatable(ax2)
# cax2=divider2.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax2)
# plt.colorbar(cax=cax2).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax2).set_label(r'\textbf{U/I}', rotation=270)
# ax3=fig1.add_subplot(spec[3,0])
# plt.imshow(v_s[:,:,50],cmap='binary_r')
# plt.xlabel("pixel")
# plt.ylabel("pixel")
# divider3=make_axes_locatable(ax3)
# cax3=divider3.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax3)
# plt.colorbar(cax=cax3).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax3).set_label(r'\textbf{V/I}', rotation=270)
# ax4=fig1.add_subplot(spec[0,1])
# plt.imshow(i1[:,:,50],cmap='binary_r')
# plt.xlabel("pixel")
# plt.ylabel("pixel")
# divider4=make_axes_locatable(ax4)
# cax4=divider4.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax4)
# plt.colorbar(cax=cax4).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax4).set_label(r'\textbf{I}', rotation=270)
# ax5=fig1.add_subplot(spec[1,1])
# plt.imshow(q1[:,:,50],cmap='binary_r',vmin=-0.0005,vmax=0.0005)
# plt.xlabel("pixel")
# plt.ylabel("pixel")
# divider5=make_axes_locatable(ax5)
# cax5=divider5.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax5)
# plt.colorbar(cax=cax5).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax5).set_label(r'\textbf{Q/I}', rotation=270)
# ax6=fig1.add_subplot(spec[2,1])
# plt.imshow(u1[:,:,50],cmap='binary_r',vmin=-0.0005,vmax=0.0005)
# plt.xlabel("pixel")
# plt.ylabel("pixel")
# divider6=make_axes_locatable(ax6)
# cax6=divider6.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax6)
# plt.colorbar(cax=cax6).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax6).set_label(r'\textbf{U/I}', rotation=270)
# ax7=fig1.add_subplot(spec[3,1])
# plt.imshow(v1[:,:,50],cmap='binary_r',vmin=-0.00016,vmax=0.00011)
# plt.xlabel("pixel")
# plt.ylabel("pixel")
# divider7=make_axes_locatable(ax7)
# cax7=divider7.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax7)
# plt.colorbar(cax=cax7).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax7).set_label(r'\textbf{V/I}', rotation=270)
# plt.savefig('PCA_calma.png')


# fig2=plt.figure(figsize=(13,13))
# fig1=fig2
# spec=gridspec.GridSpec(ncols=2,nrows=4)
# ax0=fig1.add_subplot(spec[0,0])
# plt.imshow(i_m[:,:,50],cmap='binary_r')
# plt.xlabel("pixel")
# plt.ylabel("pixel")
# divider0=make_axes_locatable(ax0)
# cax0=divider0.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax0)
# plt.colorbar(cax=cax0).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax0).set_label(r'\textbf{I}', rotation=270)
# ax1=fig1.add_subplot(spec[1,0])
# plt.imshow(q_m[:,:,50],cmap='binary_r')
# plt.xlabel("pixel")
# plt.ylabel("pixel")
# divider1=make_axes_locatable(ax1)
# cax1=divider1.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax1)
# plt.colorbar(cax=cax1).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax1).set_label(r'\textbf{Q/I}', rotation=270)
# ax2=fig1.add_subplot(spec[2,0])
# plt.imshow(u_m[:,:,50],cmap='binary_r')
# plt.xlabel("pixel")
# plt.ylabel("pixel")
# divider2=make_axes_locatable(ax2)
# cax2=divider2.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax2)
# plt.colorbar(cax=cax2).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax2).set_label(r'\textbf{U/I}', rotation=270)
# ax3=fig1.add_subplot(spec[3,0])
# plt.imshow(v_m[:,:,50],cmap='binary_r',vmin=-0.03,vmax=0.02)
# plt.xlabel("pixel")
# plt.ylabel("pixel")
# divider3=make_axes_locatable(ax3)
# cax3=divider3.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax3)
# plt.colorbar(cax=cax3).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax3).set_label(r'\textbf{V/I}', rotation=270)
# ax4=fig1.add_subplot(spec[0,1])
# plt.imshow(i2[:,:,50],cmap='binary_r')
# plt.xlabel("pixel")
# plt.ylabel("pixel")
# divider4=make_axes_locatable(ax4)
# cax4=divider4.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax4)
# plt.colorbar(cax=cax4).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax4).set_label(r'\textbf{I}', rotation=270)
# ax5=fig1.add_subplot(spec[1,1])
# plt.imshow(q2[:,:,50],cmap='binary_r',vmin=-0.00030,vmax=0.00075)
# plt.xlabel("pixel")
# plt.ylabel("pixel")
# divider5=make_axes_locatable(ax5)
# cax5=divider5.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax5)
# plt.colorbar(cax=cax5).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax5).set_label(r'\textbf{Q/I}', rotation=270)
# ax6=fig1.add_subplot(spec[2,1])
# plt.imshow(u2[:,:,50],cmap='binary_r',vmin=-0.0005,vmax=0.0005)
# plt.xlabel("pixel")
# plt.ylabel("pixel")
# divider6=make_axes_locatable(ax6)
# cax6=divider6.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax6)
# plt.colorbar(cax=cax6).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax6).set_label(r'\textbf{U/I}', rotation=270)
# ax7=fig1.add_subplot(spec[3,1])
# plt.imshow(v2[:,:,50],cmap='binary_r',vmin=-0.03,vmax=0.02)
# plt.xlabel("pixel")
# plt.ylabel("pixel")
# divider7=make_axes_locatable(ax7)
# cax7=divider7.append_axes("right",size="5%", pad=0.1)
# plt.colorbar(cax=cax7)
# plt.colorbar(cax=cax7).ax.get_yaxis().labelpad=12
# plt.colorbar(cax=cax7).set_label(r'\textbf{V/I}', rotation=270)
# plt.savefig('PCA_mancha.png')

###############################################################################

#RECORDATORIO
#Tenemos entonces todos nuestros datos normalizados y calibrado en 
#i_s,q_s,u_s,v_s y ldo1 para el sol en calma
#i_m,q_m,u_m,v_m y ldo2 para los datos de la mancha solar
#I y ldo para el atlas

#DEBEMOS GUARDAR LOS DATOS EN OTRO FITS
#Buscar como facer esto (sempre podemos gardalos independentemente)
x_s=np.zeros((4,401,401,96))
x_s[0,:,:,:]=i_s
x_s[1,:,:,:]=q_s
x_s[2,:,:,:]=u_s
x_s[3,:,:,:]=v_s

x_m=np.zeros((4,399,401,96))
x_m[0,:,:,:]=i_m
x_m[1,:,:,:]=q_m
x_m[2,:,:,:]=u_m
x_m[3,:,:,:]=v_m

hdu_s = fits.PrimaryHDU(data=x_s)
hdu_s.writeto('sol.fits',overwrite=True)
hdu_m= fits.PrimaryHDU(data=x_m)
hdu_m.writeto('mancha.fits',overwrite=True)

np.savetxt('ldo_s',ldo1)
np.savetxt('ldo_m',ldo2)