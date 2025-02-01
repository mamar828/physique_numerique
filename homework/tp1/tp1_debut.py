import scipy.constants as sp


# Numéros atomiques, composition atomique et masse molaire
eau = [(1, 0.111894, 1.008), (8, 0.888106, 15.999)]
os_compact = [(1, 0.063984, 1.008), (6, 0.278000, 12.011), (7, 0.027000, 14.007), (8, 0.410016, 15.999),
              (12, 0.002000, 24.305), (15, 0.070000, 30.974), (16, 0.002000, 32.06), (20, 0.147000, 40.078)]

# Masses volumiques
rho_eau = 1.00000
rho_os_compact = 1.85000

def densite_electronique(matiere, rho):
    "Reçoit en argument la composition chimique d'une matière et sa masse volumique en g/cm^3"
    "et retourne sa densité électronique en électrons par cm^3"
    n_e = 0
    for x in matiere:
        n_e += x[0]*x[1]/x[2]
    return rho*n_e*sp.N_A
