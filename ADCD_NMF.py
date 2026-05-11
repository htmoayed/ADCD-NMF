import torch



def ADCD_NMF(A,X, D_n, S_m, D_m, alpha,beta,gamma, lambda_, k,n, m,inneriter):
# Initialization
    torch.set_default_tensor_type(torch.DoubleTensor)

    E = torch.rand(m, k)
    G = torch.rand(k, m)


    eps = 1e-10
    #gamma = beta
    #lambda_ = alpha
    for iter in range(inneriter):
        # Intermediate: GX (K, m)
        GX = G @ X

        # Numerator (n, K)
        num = X @ GX.T + beta * S_m @ E + 2.0 * gamma * E

        # Denominator (n, K)
        # Use smaller product chains to save GPU memory:
        #   E GX (GX)^T   = E @ (GX @ GX.T)
        #   E E^T E       = E @ (E.T @ E)
        den = (E @ (GX @ GX.T)
               + beta * D_m @ E
               + 2.0 * gamma * (E @ (E.T @ E)))

        # Fourth‑root update (element‑wise)
        E = E * (num / (den + eps)).pow(0.25)
    # ---------------------------------------------------------------------------
        # Commonly reused intermediates
        XXT = X @ X.T  # (n, n)
        GX = G @ X  # (K, m)
        GXXT = G @ XXT  # (K, n)   = G @ X @ X^T

        # ---- Numerator ----
        term1_num = (E.T * G) @ XXT  # (E^T ⊙ G) @ X @ X^T
        term2_num = (2.0 + alpha) * (GX @ A @ X.T)  # G X A X^T
        term3_num = 2.0 * lambda_ * G
        numerator = term1_num + term2_num + term3_num

        # ---- Denominator ----
        term1_den = (E.T @ E) @ G @ XXT  # (E^T E) G X X^T
        term2_den = 2.0 * (GXXT ** 2)  # element‑wise square of GXXT
        term3_den = alpha * (GX @ D_n @ X.T)  # G X D_n X^T
        term4_den = 2.0 * lambda_ * (G @ G.T @ G)  # G G^T G
        denominator = term1_den + term2_den + term3_den + term4_den

        # Fourth‑root element‑wise update
        G = G * (numerator / (denominator + eps)).pow(0.25)
    # -----------------------------------------------------------------------------------
    return E, G
