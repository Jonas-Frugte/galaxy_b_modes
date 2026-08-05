import healpy as hp
import numpy as np

def der_1(a1, a2):
    return -1 * a1, -1 * a2

def der_2(kappa, gamma1, gamma2):
    """Assemble the symmetric second-derivative tensor D_ij."""
    D_tt = kappa + gamma1
    D_tp = gamma2
    D_pp = kappa - gamma1
    return D_tt, D_tp, D_pp

def flexion_to_D(F1, F2, G1, G2):
    """Assemble the symmetric third-derivative tensor D_ijk (Bacon eqs 16-17)."""
    D_ttt = -0.5 * (3 * F1 + G1)
    D_ttp = -0.5 * (F2 + G2)
    D_tpp = -0.5 * (F1 - G1)
    D_ppp = -0.5 * (3 * F2 - G2)
    return D_ttt, D_ttp, D_tpp, D_ppp