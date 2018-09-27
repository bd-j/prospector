import numpy as np

__all__ = ["convergence_check", "make_kl_bins", "kl_divergence",
           "find_subsequence"]


def find_subsequence(subseq, seq):
    """If subsequence exists in sequence, return True. otherwise return False.
    can be modified to return the appropriate index (useful to test WHERE a
    chain is converged)
    """
    i, n, m = -1, len(seq), len(subseq)
    try:
        while True:
            i = seq.index(subseq[0], i + 1, n - m + 1)
            if subseq == seq[i:i + m]:
                #return i+m-1  (could return "converged" index here)
                return True
    except ValueError:
        return False


def kl_divergence(pdf1, pdf2):
    """Calculates Kullback-Leibler (KL) divergence for two discretized PDFs
    """
    idx = (pdf1 != 0)  # no contribution from bins where there is no density in the target PDF
    pdf1 = pdf1 / float(pdf1.sum())
    pdf2 = pdf2 / float(pdf2.sum())
    dl = pdf1[idx] * np.log(pdf1[idx] / pdf2[idx])

    return dl.sum()


def make_kl_bins(chain, nbins=10):
    """Create bins with an ~equal number of data points in each when there are
    empty bins, the KL divergence is undefined this adaptive binning scheme
    avoids that problem
    """
    sorted = np.sort(chain)
    nskip = np.floor(chain.shape[0]/float(nbins)).astype(int)-1
    bins = sorted[::nskip]
    bins[-1] = sorted[-1]  # ensure the maximum bin is the maximum of the chain
    assert bins.shape[0] == nbins+1
    pdf, bins = np.histogram(chain, bins=bins)

    return pdf, bins


def convergence_check(chain, convergence_check_interval=None, convergence_chunks=325,
                      convergence_stable_points_criteria=3, convergence_nhist=50,
                      convergence_kl_threshold=0.018, **kwargs):
    """Performs a Kullback-Leibler divergence test for convergence.

    :param chain:
        The chain to perform the test on.

    :param convergence_check_interval:
        How often to assess convergence, in number of iterations.

    :param convergence_chunks:
        The number of iterations to combine when creating the marginalized
        parameter probability functions.

    :param convergence_stable_points_criteria:
        The number of stable convergence checks that the chain must pass before
        being declared stable.

    :param convergence_nhist:
        Controls how finely the PDF is subsampled before comparison. This has a
        strong effect on the normalization of the KL divergence. Larger -->
        more noise but finer distinction between PDF shapes.

    :param convergence_kl_threshold:
        The convergence criteria for the KL test. Suggest running multiple long
        chains and plotting the KL divergence in each parameter to determine
        how to set this.

    :returns convergence_flag:
        True if converged. False if not.

    :returns outdict:
        Contains the results of the KL test for each parameter (number of
        checks, number of parameters) and the iteration where this was
        calculated.
    """

    nwalkers, niter, npars = chain.shape

    # Define some useful quantities
    niter_check_start = 2*convergence_chunks # must run for at least 2 intervals before checking!
    ncheck = np.floor((niter-niter_check_start)/float(convergence_check_interval)).astype(int)+1

    # Calculate the K-L divergence in each chunk for each parameter
    kl = np.zeros(shape=(ncheck, npars))
    xiter = np.arange(ncheck) * convergence_check_interval + niter_check_start
    for n in range(ncheck):
        for i in range(npars):

            # Define chains and calculate pdf
            lo = (xiter[n] - 2*convergence_chunks)
            hi = (xiter[n] - convergence_chunks)
            early_test_chain = chain[:, lo:hi, i].flatten()
            pdf_early, bins = make_kl_bins(early_test_chain, nbins=convergence_nhist)

            # clip test chain so that it's all contained in bins
            # basically redefining first and last bin to have open edges
            late_test_chain = np.clip(chain[:, hi:xiter[n], i].flatten(),
                                      bins[0], bins[-1])
            pdf_late, _ = np.histogram(late_test_chain, bins=bins)
            kl[n, i] = kl_divergence(pdf_late, pdf_early)

    # Check for convergence
    converged_idx = np.all(kl < convergence_kl_threshold, axis=1)
    convergence_flag = find_subsequence([True]*convergence_stable_points_criteria,
                                        converged_idx.tolist())

    outdict = {'iteration': xiter, 'kl_test': kl}
    return convergence_flag, outdict
