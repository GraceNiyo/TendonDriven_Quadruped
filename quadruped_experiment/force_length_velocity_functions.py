# --- FL, FV, FP functions ---


def bump(L, A, mid, B):
    left = 0.5*(A + mid)
    right = 0.5*(mid + B)
    if L <= A or L >= B:
        return 0
    elif L < left:
        x = (L - A) / (left - A)
        return 0.5 * x * x
    elif L < mid:
        x = (mid - L) / (mid - left)
        return 1 - 0.5 * x * x
    elif L < right:
        x = (L - mid) / (right - mid)
        return 1 - 0.5 * x * x
    else:
        x = (B - L) / (B - right)
        return 0.5 * x * x

def compute_FL(L):
    lmin = 0.5
    lmax = 1.6
    return bump(L, lmin, 1, lmax) + 0.15 * bump(L, lmin, 0.5*(lmin + 0.95), 0.95)

def compute_FP(L):
    lmin = 0.5
    lmax = 1.6
    fpmax = 1.3
    a = 0.5 * (lmin + 1)
    b = 0.5 * (1 + lmax)
    if L <= 1:
        return 0
    elif L <= b:
        x = (L - 1) / (b - 1)
        return 0.25 * fpmax * x * x * x
    else:
        x = (L - b) / (b - 1)
        return 0.25 * fpmax * (1 + 3 * x)

def compute_FV(V):
    
    fvmax = 1.2
    vmax = 1.5
    c = fvmax - 1
    V_norm = V / vmax
    if V_norm <= -1:
        return 0
    elif V_norm <= 0:
        return (V_norm + 1) ** 2
    elif V_norm <= c:
        return fvmax - (c - V_norm) * (c - V_norm) / c
    else:
        return fvmax


