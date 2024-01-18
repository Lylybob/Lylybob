## import librairies
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import cartopy.crs as ccrs
import cartopy.feature as cfeature # une librairie utile pour tracer les continents, océans, et frontières
import pickle # pour enregistrer/lire des dictionnaires
import datetime
import copy
from matplotlib.lines import Line2D  # pour legend
from matplotlib.colors import to_rgba_array # pour créer liste de couleurs distinctes

## importer data
path = 'D:/Users/elise/Documents/CDD_Paris_2023-2024/OBS/obs_mosaic/'
file = 'mosaic_clean.csv'
df = pd.read_csv(path+file)

df_grouped = df.groupby('Event').agg({
    'Date': 'first',  # Prendre la première date du groupe
    'Date/Time': 'first',
    'Latitude': 'first',
    'Longitude': 'first',
    'EsEs [m]': 'first',
    'Core length [m]': 'first',
    'Snow h [m]':'last',
    'Comment' : 'first' ,
    'Mean Salinity [psu]': 'mean',
    'Mean Temperature [°C]': 'mean'
}).reset_index()

## Figure distribution cl
hi = df['EsEs [m]'].unique()
cl = df['Core length [m]'].unique()

plt.figure()
plt.hist(hi, bins=40, density=True, color='blue',  label='Mean = '+str(round(np.nanmean(hi),2))+' m', alpha=0.5)
plt.xlabel('Ice Thickness (m)')
plt.title('Ice Thickness Distribution (hi)')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

## Figure distribution cl
plt.figure()
plt.hist(cl, bins=40, density=True, color='blue',  label='Mean = '+str(round(np.nanmean(cl),2))+' m', alpha=0.5)
plt.xlabel('Ice Thickness (m)')
plt.title('Ice Thickness Distribution (Core Length)')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

## Figure distribution de Smoy et Tmoy RAW
# Salinity
Smoy = df['Mean Salinity [psu]'].unique()

plt.figure()
plt.hist(Smoy, bins=40, density=True, color='blue',  label='Mean = '+str(round(np.nanmean(Smoy),2))+' psu', alpha=0.5)
#plt.vlines(3, 0, 0.5, color='blue')
plt.vlines(4, 0, 1, color='black')
plt.vlines(5.5, 0, 1)
plt.xlabel('Mean Salinity (psu)')
plt.title('Ice Mean Raw Salinity Distribution')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# Temperature
Tmoy = df['Mean Temperature [°C]'].unique()

plt.figure()
plt.hist(Tmoy, bins=40, density=True, color='blue',  label='Mean = '+str(round(np.nanmean(Tmoy),2))+' degC', alpha=0.5)
plt.vlines(-1.8, 0, 0.5)
plt.vlines(-5, 0, 0.5)
plt.xlabel('Mean Temperature (degC)')
plt.title('Ice Mean Raw Temperature Distribution')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

## Figure Smoyen = f(Tmoyen)
plt.figure()
plt.scatter(Tmoy, Smoy, color='blue', alpha=0.5, label='Mosaic')
plt.xlabel('Mean Raw Temperature (degC)')
plt.ylabel('Mean Raw Salinity (psu)')
plt.legend()
plt.tight_layout()
plt.show()

## Figure Smoyen = f (thickness)
comments = df_grouped['Comment']
# Kovacs
def kovacs(h):
    return(4.606 + 0.91603/h)

x_thickness = np.arange(0.07,3.5, 0.01)
S_kovacs = kovacs(x_thickness)

# curve fit
from scipy.optimize import curve_fit
def kov_fit(x, a, b):
    return a + b/x
xdata = []
ydata = []

# épaisseur = cl
plt.figure(figsize=(12,7))

for T, cli, S, comment in zip(Tmoy, cl, Smoy, comments) :
    print(comment)
    if (T > -1.8) or (comment != None): #ete, false bot or rafted
        plt.scatter(cli, S, color='steelblue', alpha=0.5)
    else :
        plt.scatter(cli, S, color='blue', alpha=0.5)
        xdata.append(cli)
        ydata.append(S)

popt, pcov = curve_fit(kov_fit, xdata, ydata, p0 = [0,0]) #[4.606, 0.91603]
perr = np.sqrt(np.diag(pcov))
a, b = popt[0], popt[1]
a_std, b_std = perr[0], perr[1]
yfit = kov_fit(x_thickness, a, b)
yfit_dwn = kov_fit(x_thickness, a+2*a_std, b+2*b_std)
yfit_up = kov_fit(x_thickness, a-2*a_std, b-2*b_std)

plt.plot(x_thickness, yfit, color='tab:blue', linewidth=2)
plt.fill_between(x_thickness, yfit_up, yfit_dwn, color='tab:blue', alpha=.25)

plt.plot(x_thickness, S_kovacs, color='black', label='kovacs')
plt.xlabel('Core Length best estimate (m)')
plt.ylabel('Salinity (psu)')

# Création de la légende personnalisée
legend_elements = []
for color, season in zip(['blue', 'steelblue'], ['winter', 'summer']):
    legend_elements.append(Line2D([0], [0], marker='o', color=color, markerfacecolor=color, markersize=10, linestyle='', label=season, alpha=0.5))
legend_elements.append(Line2D([0], [0], color='black', linestyle='solid', label='kovacs : a=4.606, b=0.916'))
legend_elements.append(Line2D([0], [0], color='tab:blue', linestyle='solid', linewidth=2, label='fit: a=%5.3f, b=%5.3f'%tuple(popt)))
legend_elements.append(Line2D([0], [0], color='tab:blue', alpha = 0.25, linestyle='', label='2-sigma interval (95%)'))

plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()

## Figure Smoyen = f(hs, Tmoyen)
# épaisseur = cl
plt.figure(figsize=(10,7))
plt.scatter(Tmoy, cl,  c=Smoy, alpha=1, s=50, cmap='Oranges')
plt.ylabel('Core length (m)')
plt.xlabel('Temperature (degC)')
plt.legend()
plt.colorbar(label='Salinity (psu)')
plt.tight_layout()
plt.show()

# épaisseur = hi
plt.figure(figsize=(10,7))
plt.scatter(Tmoy, hi,  c=Smoy, alpha=1, s=50, cmap='Oranges')
plt.ylabel('hi (m)')
plt.xlabel('Temperature (degC)')
plt.legend()
plt.colorbar(label='Salinity (psu)')
plt.tight_layout()
plt.show()
