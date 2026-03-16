import numpy as np
import sklearn
from sklearn.metrics.pairwise import polynomial_kernel

# You are allowed to import any submodules of sklearn e.g. metrics.pairwise to construct kernel Gram matrices
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_kernel, my_decode etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
################################
# Non Editable Region Starting #
################################
def my_kernel(X1, Z1, X2, Z2):
    ################################
    #  Non Editable Region Ending  #
    ################################

    # Use this method to compute Gram matrices for your proposed kernel
    # Your kernel matrix will be used to train a kernel ridge regressor

    # ---- Hyperparameters to tune ----
    c = 0.106
    d= 2
    # ---------------------------------
    # Compute the base polynomial kernel on z
    Kz = (Z1 @ Z2.T + c) ** d   # (n1, n2)

    # Compute dot product between x’s
    Kx = X1 @ X2.T              # (n1, n2)

    # Combine to form the final semi-parametric kernel
    G = Kx * Kz + 1.0           # (n1, n2)

    return G



################################
# Non Editable Region Starting #
################################


def my_decode(w):
    
################################
#  Non Editable Region Ending  #
################################

	# Use this method to invert a PUF linear model to get back delays
	# w is a single 1089-dim vector (last dimension being the bias term)
	# The output should be eight 32-dimensional vectors
    w = np.asarray(w).reshape(-1)
    k = 32
    L = k + 1

    # reshape into 33x33 matrix
    W = w.reshape(L, L)

    # since we are assuming u[0] = 1 in our solution hence v can be assumed to be the first row
    v = W[0, :]

    # we are picking the maximum value index, this is because dividing by the maximum value helps us to minimize variation and errors due to division by very small values
    idx = np.argmax(np.abs(v))
    if v[idx] == 0:
        # since we are going to divide by v[idx] to get the remaining elements of u, we will check for it being zero
        v[idx] = 1e-12

    # recovering u from column idx, as stated in the proposed solution
    u = W[:, idx] / v[idx]

    # enforce reference constraint u0 = 1, the basis of our solution
    u[0] = 1.0

    # assigning alpha and beta values by simply putting all but the last beta value as zero and the alpha values equal to the corresponding u values
    alpha_u = u[:k]
    beta_u = np.zeros(k)
    beta_u[-1] = u[-1]

    alpha_v = v[:k]
    beta_v = np.zeros(k)
    beta_v[-1] = v[-1]

    ud1 = alpha_u + beta_u
    vd1 = alpha_u - beta_u
    a = np.maximum(ud1, 0)
    b = np.maximum(-ud1, 0)
    c = np.maximum(vd1, 0)
    d = np.maximum(-vd1, 0)
    
    ud2 = alpha_v + beta_v
    vd2 = alpha_v - beta_v
    p = np.maximum(ud2, 0)
    q = np.maximum(-ud2, 0)
    r = np.maximum(vd2, 0)
    s = np.maximum(-vd2, 0)


    return a, b, c, d, p, q, r, s





