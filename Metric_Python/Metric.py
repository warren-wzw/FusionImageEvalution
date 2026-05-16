import numpy as np
from scipy.signal import convolve2d, fftconvolve
from Qabf import get_Qabf
from Nabf import get_Nabf
import math
from ssim import ssim, ms_ssim


def EN_function(image_array):
    histogram, bins = np.histogram(image_array, bins=256, range=(0, 255)) #computes the histogram
    histogram = histogram / float(np.sum(histogram))#normalizes
    entropy = -np.sum(histogram * np.log2(histogram + 1e-7))
    return entropy


def SF_function(image):
    image_array = np.array(image)
    RF = np.diff(image_array, axis=0)
    RF1 = np.sqrt(np.mean(np.mean(RF ** 2)))
    CF = np.diff(image_array, axis=1)
    CF1 = np.sqrt(np.mean(np.mean(CF ** 2)))
    SF = np.sqrt(RF1 ** 2 + CF1 ** 2)
    return SF


def SD_function(image_array):
    m, n = image_array.shape
    u = np.mean(image_array)
    SD = np.sqrt(np.sum(np.sum((image_array - u) ** 2)) / (m * n))
    return SD


def PSNR_function(A, B, F):
    A = A / 255.0
    B = B / 255.0
    F = F / 255.0
    m, n = F.shape
    MSE_AF = np.sum(np.sum((F - A) ** 2)) / (m * n)
    MSE_BF = np.sum(np.sum((F - B) ** 2)) / (m * n)
    MSE = 0.5 * MSE_AF + 0.5 * MSE_BF
    PSNR = 20 * np.log10(255 / np.sqrt(MSE))
    return PSNR


def MSE_function(A, B, F):
    A = A / 255.0
    B = B / 255.0
    F = F / 255.0
    m, n = F.shape
    MSE_AF = np.sum(np.sum((F - A) ** 2)) / (m * n)
    MSE_BF = np.sum(np.sum((F - B) ** 2)) / (m * n)
    MSE = 0.5 * MSE_AF + 0.5 * MSE_BF
    return MSE


def fspecial_gaussian(shape, sigma):
    """
    2D gaussian mask - should give the same result as MATLAB's fspecial('gaussian',...)
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def vifp_mscale(ref, dist):
    sigma_nsq = 2
    num = 0
    den = 0
    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        win = fspecial_gaussian((N, N), N / 5)

        if scale > 1:
            ref = fftconvolve(ref, win, mode='valid')
            dist = fftconvolve(dist, win, mode='valid')
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = fftconvolve(ref, win, mode='valid')
        mu2 = fftconvolve(dist, win, mode='valid')
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = fftconvolve(ref * ref, win, mode='valid') - mu1_sq
        sigma2_sq = fftconvolve(dist * dist, win, mode='valid') - mu2_sq
        sigma12 = fftconvolve(ref * dist, win, mode='valid') - mu1_mu2
        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        g = sigma12 / (sigma1_sq + 1e-10)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq < 1e-10] = 0
        sv_sq[sigma1_sq < 1e-10] = sigma2_sq[sigma1_sq < 1e-10]
        sigma1_sq[sigma1_sq < 1e-10] = 0

        g[sigma2_sq < 1e-10] = 0
        sv_sq[sigma2_sq < 1e-10] = 0

        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= 1e-10] = 1e-10

        num += np.sum(np.log10(1 + g ** 2 * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))
    vifp = num / den
    return vifp


def VIF_function(A, B, F):
    VIF = vifp_mscale(A, F) + vifp_mscale(B, F)
    return VIF


def CC_function(A, B, F):
    rAF = np.sum((A - np.mean(A)) * (F - np.mean(F))) / np.sqrt(
        np.sum((A - np.mean(A)) ** 2) * np.sum((F - np.mean(F)) ** 2))
    rBF = np.sum((B - np.mean(B)) * (F - np.mean(F))) / np.sqrt(
        np.sum((B - np.mean(B)) ** 2) * np.sum((F - np.mean(F)) ** 2))
    CC = np.mean([rAF, rBF])
    return CC


def corr2(a, b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    r = np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b))
    return r


def SCD_function(A, B, F):
    r = corr2(F - B, A) + corr2(F - A, B)
    return r


def Qabf_function(A, B, F):
    return get_Qabf(A, B, F)


def Nabf_function(A, B, F):
    return Nabf_function(A, B, F)


def Hab(im1, im2, gray_level):
    N = gray_level
    h = np.zeros((N, N))
    np.add.at(h, (im1.ravel(), im2.ravel()), 1)
    h = h / np.sum(h)
    im1_marg = np.sum(h, axis=1)
    im2_marg = np.sum(h, axis=0)
    mask_x = im1_marg > 0
    mask_y = im2_marg > 0
    mask_xy = h > 0
    H_x = np.sum(im1_marg[mask_x] * np.log2(im1_marg[mask_x]))
    H_y = np.sum(im2_marg[mask_y] * np.log2(im2_marg[mask_y]))
    H_xy = np.sum(h[mask_xy] * np.log2(h[mask_xy]))
    return H_xy - H_x - H_y


def MI_function(A, B, F, gray_level=256):
    MIA = Hab(A, F, gray_level)
    MIB = Hab(B, F, gray_level)
    MI_results = MIA + MIB
    return MI_results


def MI_function2(A, B, C, F, gray_level=256):
    MIA = Hab(A, F, gray_level)
    MIB = Hab(B, F, gray_level)
    MIC = Hab(C, F, gray_level)
    MI_results = MIA + MIB + MIC
    return MI_results


def AG_function(image):
    width = image.shape[0]
    width = width - 1
    height = image.shape[1]
    height = height - 1
    tmp = 0.0
    [grady, gradx] = np.gradient(image)
    s = np.sqrt((np.square(gradx) + np.square(grady)) / 2)
    AG = np.sum(np.sum(s)) / (width * height)
    return AG


def SSIM_function(A, B, F):
    ssim_A = ssim(A, F)
    ssim_B = ssim(B, F)
    SSIM = ssim_B/2 + ssim_A/2
    return SSIM.item()


def MS_SSIM_function(A, B, F):
    ssim_A = ms_ssim(A, F)
    ssim_B = ms_ssim(B, F)
    MS_SSIM = ssim_B/2 + ssim_A/2
    return MS_SSIM.item()


def Nabf_function(A, B, F):
    Nabf = get_Nabf(A, B, F)
    return Nabf
