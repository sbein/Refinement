import sys

import torch
from torch import nn

import numpy as np

import matplotlib as mpl
# mpl.use('Agg')  # using Agg backend (without X-server running)  # TODO: uncomment
import matplotlib.pyplot as plt


def MMD(x, y, kernel, device):
    """Emprical maximum mean discrepancy. The lower the result, the more evidence that distributions are the same.
    from: https://www.kaggle.com/onurtunali/maximum-mean-discrepancy

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":

        bandwidth_range = [1, 2, 5, 10, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)

    return torch.mean(XX + YY - 2. * XY)


class MMD_loss(nn.Module):
    """Implementation of the empirical estimate of the maximum mean discrepancy

    from https://github.com/ZongxianLee/MMD_Loss.Pytorch/blob/master/mmd_loss.py
    see https://en.wikipedia.org/wiki/Kernel_embedding_of_distributions#Measuring_distance_between_distributions

    the MMD values for n="kernel_num" Gaussian kernels with different bandwidths are added
    "bandwidth" is the central bandwidth (=2*sigma^2 for standard Gaussian)
    the bandwidths range from "bandwidth" / ("kernel_mul" * "kernel_num" // 2) to "bandwidth" * ("kernel_mul" * "kernel_num" // 2)
    "bandwidth" can be fixed or is calculated from the L2-distance

    """
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None, one_sided_bandwidth=False, calculate_sigma_without_parameters=False):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.one_sided_bandwidth = one_sided_bandwidth
        self.calculate_sigma_without_parameters = calculate_sigma_without_parameters  # fix_sigma will be overwritten
        return

    @staticmethod
    def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None, one_sided_bandwidth=False, l2dist_out=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)

        if l2dist_out is not None:
            globals()[l2dist_out] = (torch.sum(L2_distance.data) / (n_samples**2-n_samples)).item()

        if fix_sigma:
            bandwidth = torch.tensor(fix_sigma)
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        if one_sided_bandwidth:
            bandwidth /= kernel_mul ** (kernel_num - 1)
        else:
            bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

        # print('')
        # # print(total)
        # print(total0.size())
        # # print(total0)
        # print(total1.size())
        # # print(total1)
        # # print(L2_distance)
        # print(torch.sum(L2_distance.data) / (n_samples**2-n_samples))
        # print(bandwidth_list)
        # # print(kernel_val)
        # print(L2_distance)
        # print([-L2_distance / bandwidth_temp for bandwidth_temp in bandwidth_list])
        # print(kernel_val)
        # print(sum(kernel_val))
        # batch_size = int(source.size()[0])
        # print([torch.mean(k[:batch_size, :batch_size] + k[batch_size:, batch_size:] - k[:batch_size, batch_size:] - k[batch_size:, :batch_size]) for k in kernel_val])

        return sum(kernel_val)

    def forward(self, source, target, parameters=None, mask=None, l2dist_out=None):

        if self.calculate_sigma_without_parameters:

            n_samples = int(source.size()[0])+int(target.size()[0])

            if mask is not None:
                total = torch.cat([source[mask], target[mask]], dim=0)
            else:
                total = torch.cat([source, target], dim=0)

            total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
            total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
            L2_distance = ((total0-total1)**2).sum(2)

            sigma = torch.sum(L2_distance.data) / (n_samples**2-n_samples)

        else:

            sigma = self.fix_sigma

        if parameters is not None:
            source = torch.cat((source, parameters), dim=1)
            target = torch.cat((target, parameters), dim=1)

        if mask is not None:
            source = source[mask]
            target = target[mask]

        # print('')
        # print('source')
        # print(source.size())
        # print(source)
        # print('target')
        # print(target.size())
        # print(target)

        batch_size = int(source.size()[0])
        kernels = MMD_loss.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=sigma, one_sided_bandwidth=self.one_sided_bandwidth, l2dist_out=l2dist_out)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)

        # print(batch_size)
        # print(kernels)
        # print(XX)
        # print(YY)
        # print(XY)
        # print(YX)
        # print(XX + YY - XY - YX)
        # print(loss)
        # print('')

        return loss


def testMMD(tBoth):

    import scipy.stats
    from my_modules import OneHotEncode

    isrefined = True
    addgradient = False
    logittransform = False

    # batchsizes = np.linspace(10, 1000, 100)
    # batchsizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    batchsizes = [512, 1024, 2048, 4096]
    # batchsizes = [512]  # TODO: remove

    samples = 21  # 51
    # samples = 3  # TODO: remove

    # bandwidths = np.logspace(-2., 4., num=13)
    # bandwidths = np.logspace(-3., 3., num=7)
    bandwidths = np.logspace(-4., 2., num=13)
    # bandwidths = np.logspace(-4., 2., num=7)
    # bandwidths = [0.01, 0.1, 1., 10.]  # TODO: remove

    # to test the MMD loss
    full_list = {}
    fast_list = {}
    param_list = {}
    full = {}
    fast = {}
    param = {}

    for offset in range(samples):
        for bs in batchsizes:
            full_list[bs + 0.01 * offset] = []
            fast_list[bs + 0.01 * offset] = []
            param_list[bs + 0.01 * offset] = []
            for ijet, jet in enumerate(tBoth):

                if ijet < offset * max(batchsizes): continue

                # if jet.JetDrGenRec_FastSim < 0: continue
                # if jet.JetDrGenRec_FullSim < 0: continue

                # full_list.append([jet.RecJetEta_FullSim])
                # fast_list.append([jet.RecJetEta_FastSim])

                # full_list.append([(jet.RecJetPt_FullSim-jet.GenJetPt)/jet.GenJetPt])
                # fast_list.append([(jet.RecJetPt_FastSim-jet.GenJetPt)/jet.GenJetPt])

                # full_list.append([(jet.RecJetPt_FullSim-jet.GenJetPt)/jet.GenJetPt, jet.RecJetEta_FullSim])
                # fast_list.append([(jet.RecJetPt_FastSim-jet.GenJetPt)/jet.GenJetPt, jet.RecJetEta_FastSim])

                # full_list[bs].append([jet.JetResponseFullSim, jet.RecJetEta_FullSim, jet.RecJetDeepCsv_FullSim])
                # fast_list[bs].append([jet.JetResponseFastSim, jet.RecJetEta_FastSim, jet.RecJetDeepCsv_FastSim])

                # full_list[bs + 0.01 * offset].append([jet.RecJetPt_FullSim, jet.RecJetEta_FullSim, jet.RecJetDeepCsv_FullSim])
                # fast_list[bs + 0.01 * offset].append([jet.RecJetPt_FastSim, jet.RecJetEta_FastSim, jet.RecJetDeepCsv_FastSim])

                if isrefined:
                    full_list[bs + 0.01 * offset].append([jet.RecJet_btagDeepFlavB_Refined, jet.RecJet_btagDeepFlavCvB_Refined, jet.RecJet_btagDeepFlavCvL_Refined, jet.RecJet_btagDeepFlavQG_Refined])
                else:
                    full_list[bs + 0.01 * offset].append([jet.RecJet_btagDeepFlavB_FullSim, jet.RecJet_btagDeepFlavCvB_FullSim, jet.RecJet_btagDeepFlavCvL_FullSim, jet.RecJet_btagDeepFlavQG_FullSim])
                fast_list[bs + 0.01 * offset].append([jet.RecJet_btagDeepFlavB_FastSim, jet.RecJet_btagDeepFlavCvB_FastSim, jet.RecJet_btagDeepFlavCvL_FastSim, jet.RecJet_btagDeepFlavQG_FastSim])
                param_list[bs + 0.01 * offset].append([jet.GenJet_pt, jet.GenJet_eta, jet.RecJet_hadronFlavour_FastSim])

                if len(full_list[bs + 0.01 * offset]) == bs: break

            # transformations:
            if logittransform:
                full[bs + 0.01 * offset] = torch.logit(torch.tensor(full_list[bs + 0.01 * offset], requires_grad=addgradient))
                fast[bs + 0.01 * offset] = torch.logit(torch.tensor(fast_list[bs + 0.01 * offset], requires_grad=addgradient))
                # param[bs + 0.01 * offset] = torch.cat((torch.tensor(param_list[bs + 0.01 * offset])[:, 0], torch.tensor(param_list[bs + 0.01 * offset])[:, 1], torch.tensor(param_list[bs + 0.01 * offset])[:, 2]), dim=1)
                param[bs + 0.01 * offset] = torch.cat((torch.unsqueeze(torch.logit(torch.tanh(torch.tensor(param_list[bs + 0.01 * offset], requires_grad=addgradient)[:, 0] / 200.)), 1),
                                                       torch.unsqueeze(torch.tensor(param_list[bs + 0.01 * offset], requires_grad=addgradient)[:, 1], 1),
                                                       OneHotEncode(source_idx=0, target_vals=[0, 4, 5]).forward(torch.unsqueeze(torch.tensor(param_list[bs + 0.01 * offset], requires_grad=addgradient)[:, 2], 1))
                                                       ), dim=1)
            else:
                full[bs + 0.01 * offset] = torch.tensor(full_list[bs + 0.01 * offset], requires_grad=addgradient)
                fast[bs + 0.01 * offset] = torch.tensor(fast_list[bs + 0.01 * offset], requires_grad=addgradient)
                # param[bs + 0.01 * offset] = torch.cat((torch.tensor(param_list[bs + 0.01 * offset])[:, 0], torch.tensor(param_list[bs + 0.01 * offset])[:, 1], torch.tensor(param_list[bs + 0.01 * offset])[:, 2]), dim=1)
                param[bs + 0.01 * offset] = torch.cat((torch.unsqueeze(torch.tanh(torch.tensor(param_list[bs + 0.01 * offset], requires_grad=addgradient)[:, 0] / 200.), 1),
                                                       torch.unsqueeze(torch.tensor(param_list[bs + 0.01 * offset], requires_grad=addgradient)[:, 1], 1),
                                                       OneHotEncode(source_idx=0, target_vals=[0, 4, 5]).forward(torch.unsqueeze(torch.tensor(param_list[bs + 0.01 * offset], requires_grad=addgradient)[:, 2], 1))
                                                       ), dim=1)

            # print(full[bs].shape)
            # print(fast[bs].shape)

    listFastFull = {}
    listFullFull = {}
    listFastFullSameGEN = {}
    listFastFullL2dist = {bs: [] for bs in batchsizes}
    listFullFullL2dist = {bs: [] for bs in batchsizes}
    listFastFullL2distSameGEN = {bs: [] for bs in batchsizes}
    grads = {}
    for bw in bandwidths:
        loss = MMD_loss(kernel_mul=1., kernel_num=1, fix_sigma=bw, one_sided_bandwidth=False)
        listFastFull[bw] = {}
        listFullFull[bw] = {}
        listFastFullSameGEN[bw] = {}
        grads[bw] = {}
        for bs in batchsizes:
            listFastFull[bw][bs] = []
            listFullFull[bw][bs] = []
            listFastFullSameGEN[bw][bs] = []
            for offset in range(samples-1):

                print(bw, bs, offset)

                valFastFull = loss.forward(full[bs + 0.01 * offset], fast[bs + 0.01 * (offset + 1)]).item()
                valFullFull = loss.forward(full[bs + 0.01 * offset], full[bs + 0.01 * (offset + 1)]).item()
                valFastFullSameGEN = loss.forward(torch.cat((full[bs + 0.01 * offset], param[bs + 0.01 * offset]), dim=1),
                                                  torch.cat((fast[bs + 0.01 * offset], param[bs + 0.01 * offset]), dim=1)).item()

                if addgradient and bs == batchsizes[-1]:
                    fast[bs + 0.01 * offset].retain_grad()
                    theloss = loss(torch.cat((full[bs + 0.01 * offset], param[bs + 0.01 * offset]), dim=1),
                                   torch.cat((fast[bs + 0.01 * offset], param[bs + 0.01 * offset]), dim=1))
                    grads[bw][bs + 0.01 * offset] = torch.autograd.grad(theloss, (fast[bs + 0.01 * offset]))[0]

                # print(theloss)
                # print(torch.autograd.grad(theloss, (fast[bs + 0.01 * offset])))  # , torch.ones_like(theloss)
                # print(theloss.sum().backward())
                # print(fast[bs + 0.01 * offset].grad)

                total = torch.cat([full[bs + 0.01 * offset], fast[bs + 0.01 * (offset + 1)]], dim=0)
                total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
                total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
                L2_distance = ((total0-total1)**2).sum(2)
                l2FastFull = torch.sum(L2_distance.data) / ((2 * bs)**2-(2 * bs))

                total = torch.cat([full[bs + 0.01 * offset], full[bs + 0.01 * (offset + 1)]], dim=0)
                total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
                total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
                L2_distance = ((total0-total1)**2).sum(2)
                l2FullFull = torch.sum(L2_distance.data) / ((2 * bs)**2-(2 * bs))

                total = torch.cat([torch.cat((full[bs + 0.01 * offset], param[bs + 0.01 * offset]), dim=1), torch.cat((fast[bs + 0.01 * offset], param[bs + 0.01 * offset]), dim=1)], dim=0)
                total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
                total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
                L2_distance = ((total0-total1)**2).sum(2)
                l2FastFullSameGEN = torch.sum(L2_distance.data) / ((2 * bs)**2-(2 * bs))

                listFastFull[bw][bs].append(valFastFull)
                listFullFull[bw][bs].append(valFullFull)
                listFastFullSameGEN[bw][bs].append(valFastFullSameGEN)

                listFastFullL2dist[bs].append(l2FastFull)
                listFullFullL2dist[bs].append(l2FullFull)
                listFastFullL2distSameGEN[bs].append(l2FastFullSameGEN)

            listFastFull[bw][bs] = np.mean(listFastFull[bw][bs])
            listFullFull[bw][bs] = np.mean(listFullFull[bw][bs])
            listFastFullSameGEN[bw][bs] = np.mean(listFastFullSameGEN[bw][bs])

    for bs in batchsizes:
        listFastFullL2dist[bs] = np.mean(listFastFullL2dist[bs])
        listFullFullL2dist[bs] = np.mean(listFullFullL2dist[bs])
        listFastFullL2distSameGEN[bs] = np.mean(listFastFullL2distSameGEN[bs])

    print(listFastFull)
    print(listFullFull)
    print(listFastFullSameGEN)
    print(listFastFullL2dist)
    print(listFullFullL2dist)
    print(listFastFullL2distSameGEN)


    labelsize = 20
    ticklabelsize = 20
    legendsize = 20
    linewidth = 2
    markersize = 10

    for i in range(2):

        fig, ax = plt.subplots(figsize=(20, 10))

        l2distFastFullsum = 0
        l2distFullFullsum = 0
        l2distFastFullsumSameGEN = 0
        for bs in batchsizes:
            if i == 0:
                p = ax.plot(bandwidths, [listFastFull[bw][bs] for bw in bandwidths], linewidth=linewidth, linestyle='-', label='Fast/Full (different GEN), BS=' + str(bs))
                ax.plot(bandwidths, [listFullFull[bw][bs] for bw in bandwidths], linewidth=linewidth, linestyle=':', color=p[0].get_color(), label='Full/Full (different GEN), BS=' + str(bs))
            else:
                ax.plot(bandwidths, [bs * listFastFullSameGEN[bw][bs] for bw in bandwidths], linewidth=linewidth, linestyle='-.', label=str(bs) + ' * Fast/Full+Param. (same GEN), BS=' + str(bs))  # , color=p[0].get_color()
            l2distFastFullsum += listFastFullL2dist[bs]
            l2distFullFullsum += listFullFullL2dist[bs]
            l2distFastFullsumSameGEN += listFastFullL2distSameGEN[bs]

        if i == 0:
            ax.axvline(l2distFastFullsum / len(batchsizes), linewidth=linewidth, linestyle='-', color='black', label='L2dist Fast/Full (different GEN)')
            ax.axvline(l2distFullFullsum / len(batchsizes), linewidth=linewidth, linestyle=':', color='black', label='L2dist Full/Full (different GEN)')
        else:
            ax.axvline(l2distFastFullsumSameGEN / len(batchsizes), linewidth=linewidth, linestyle='-.', color='black', label='L2dist Fast/Full+Param. (same GEN)')


        ax.legend(prop={'size': legendsize})

        ax.set_xscale('log')
        # if i == 1: ax.set_yscale('log')

        ax.set_xlabel('Bandwidth', size=labelsize)
        ax.set_ylabel('MMD', size=labelsize)

        ax.tick_params(axis='x', labelsize=ticklabelsize)
        ax.tick_params(axis='y', labelsize=ticklabelsize)

        fig.tight_layout()
        fig.savefig('/afs/desy.de/user/w/wolfmor/Plots/Refinement/Training/Regression/MMD/MMDbandwidthFastFullTest' + ('_diffGEN' if i == 0 else '_sameGEN') + ('_refined20230218_1' if isrefined else '') + '.png')

    if addgradient:

        # TODO: adapt to refined?

        bins_logit = np.linspace(-10, 10, 40)
        bins = np.linspace(0, 1, 40)
        bincenters = np.multiply(0.5, bins[1:] + bins[:-1])
        for bw in bandwidths:

            for bs in batchsizes[-1:]:

                fig, ax = plt.subplots(3, 4, figsize=(4*10, 2*10))
                addax = {}
                for idim, dim in enumerate(['B', 'CvB', 'CvL', 'QG']):

                    fastacc = torch.cat([fast[bs + 0.01 * offset][:, idim].detach() for offset in range(samples-1)])
                    fullacc = torch.cat([full[bs + 0.01 * offset][:, idim].detach() for offset in range(samples-1)])

                    if logittransform:
                        ax[0, idim].hist(torch.unsqueeze(fastacc, 0), bins=bins_logit, color='red', histtype='step', linewidth=linewidth, label='Fast')
                        ax[0, idim].hist(torch.unsqueeze(fullacc, 0), bins=bins_logit, color='green', histtype='step', linewidth=linewidth, label='Full')
                    else:
                        ax[0, idim].hist(torch.unsqueeze(torch.logit(fastacc), 0), bins=bins_logit, color='red', histtype='step', linewidth=linewidth, label='Fast')
                        ax[0, idim].hist(torch.unsqueeze(torch.logit(fullacc), 0), bins=bins_logit, color='green', histtype='step', linewidth=linewidth, label='Full')

                    ax[0, idim].legend(prop={'size': legendsize})

                    ax[0, idim].set_xlabel('logit(' + dim + ')', size=labelsize)
                    ax[0, idim].set_ylabel('Entries', size=labelsize)

                    ax[0, idim].tick_params(axis='x', labelsize=ticklabelsize)
                    ax[0, idim].tick_params(axis='y', labelsize=ticklabelsize)

                    if logittransform:
                        hfast, _, _ = ax[1, idim].hist(torch.unsqueeze(torch.sigmoid(fastacc), 0), bins=bins, color='red', histtype='step', linewidth=linewidth, label='Fast')
                        hfull, _, _ = ax[1, idim].hist(torch.unsqueeze(torch.sigmoid(fullacc), 0), bins=bins, color='green', histtype='step', linewidth=linewidth, label='Full')
                    else:
                        hfast, _, _ = ax[1, idim].hist(torch.unsqueeze(fastacc, 0), bins=bins, color='red', histtype='step', linewidth=linewidth, label='Fast')
                        hfull, _, _ = ax[1, idim].hist(torch.unsqueeze(fullacc, 0), bins=bins, color='green', histtype='step', linewidth=linewidth, label='Full')

                    ax[1, idim].legend(prop={'size': legendsize})

                    ax[1, idim].set_xlabel(dim, size=labelsize)
                    ax[1, idim].set_ylabel('Entries', size=labelsize)

                    ax[1, idim].tick_params(axis='x', labelsize=ticklabelsize)
                    ax[1, idim].tick_params(axis='y', labelsize=ticklabelsize)


                    # gradients:

                    x = np.array([])
                    y = np.array([])
                    for offset in range(samples-1):
                        if logittransform:
                            ax[2, idim].plot(torch.sigmoid(fast[bs + 0.01 * offset][:, idim]).detach().numpy(), grads[bw][bs + 0.01 * offset][:, idim], '*', color='lightgray', zorder=-1)
                            x = np.concatenate((x, torch.sigmoid(fast[bs + 0.01 * offset][:, idim]).detach().numpy()))
                            y = np.concatenate((y, grads[bw][bs + 0.01 * offset][:, idim]))
                        else:
                            ax[2, idim].plot(fast[bs + 0.01 * offset][:, idim].detach().numpy(), grads[bw][bs + 0.01 * offset][:, idim], '*', color='lightgray', zorder=-1)
                            x = np.concatenate((x, fast[bs + 0.01 * offset][:, idim].detach().numpy()))
                            y = np.concatenate((y, grads[bw][bs + 0.01 * offset][:, idim]))

                    means_result = scipy.stats.binned_statistic(x, [y, y**2], bins=bins, statistic='mean')
                    means, means2 = means_result.statistic
                    standard_deviations = np.sqrt(means2 - means**2)

                    ax[2, idim].errorbar(bincenters, means, yerr=standard_deviations, label='Gradients profile', fmt='_', color='blue', linewidth=2*linewidth, markeredgewidth=2*linewidth, markersize=markersize)  # , zorder=2.5


                    # ratios:

                    mask = np.logical_and(hfast != 0, hfull != 0)
                    ratio = np.divide(hfast, hfull, where=mask)
                    error = ratio * np.sqrt(np.divide(1, hfast, where=mask) + np.divide(1, hfull, where=mask))

                    addax[idim] = ax[2, idim].twinx()
                    addax[idim].errorbar(bincenters, ratio, yerr=error, fmt='_', color='red', linewidth=linewidth, markeredgewidth=linewidth, markersize=markersize)  # , zorder=2.4
                    addax[idim].axhline(1., linewidth=linewidth, linestyle='dotted', color='red')
                    addax[idim].set_ylim((0.7, 1.3))
                    addax[idim].set_ylabel('Fast / Full', size=labelsize)
                    addax[idim].tick_params(axis='y', labelsize=ticklabelsize)

                    ax[2, idim].legend(loc='upper right', prop={'size': legendsize})

                    ax[2, idim].set_xlim(ax[1, idim].get_xlim())
                    ylim = max([abs(yl) for yl in ax[2, idim].get_ylim()])
                    ax[2, idim].set_ylim((-ylim, ylim))

                    ax[2, idim].set_xlabel(dim, size=labelsize)
                    ax[2, idim].set_ylabel(r'$\partial MMD \, / \, \partial ' + dim + '$', size=labelsize)

                    ax[2, idim].tick_params(axis='x', labelsize=ticklabelsize)
                    ax[2, idim].tick_params(axis='y', labelsize=ticklabelsize)

                    ax[2, idim].yaxis.offsetText.set_fontsize(ticklabelsize)

                fig.tight_layout()
                fig.savefig('/afs/desy.de/user/w/wolfmor/Plots/Refinement/Training/Regression/MMD/MMDgrads_BS' + str(bs) + '_BW' + str(bw).replace('.', 'p') + '.png')


    sys.exit(0)

    loss = MMD_loss(kernel_mul=10., kernel_num=5)
    loss_list = {}
    for offset in range(samples):
        loss_list[offset] = []
        for bs in batchsizes:
            val = loss.forward(full[bs + 0.01 * offset], fast[bs + 0.01 * offset]).item()
            loss_list[offset].append(val)
            # print(bs, 'loss', val)

    # for sgm in np.arange(1, 5, step=0.5):
    #     loss = MMD_loss(kernel_mul=2., kernel_num=5, fix_sigma=sgm)
    #     val = loss.forward(full, fast).item()
    #     loss_list.append(val)
    #     print(sgm, 'loss', val)

    # print(batchsizes)
    # print(loss_list)

    for offset in range(samples):
        plt.plot(batchsizes, loss_list[offset])

    plt.xlabel('batchsize')
    plt.ylabel('MMD')

    plt.show()

    # # kernel_mul_list = [1., 2., 3., 4., 5., 6., 7., 8., 9.]
    # # kernel_num_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # kernel_mul_list = np.arange(1., 11.)
    # kernel_num_list = np.arange(1, 11)
    # loss_list = []
    #
    # for ikmul, kmul in enumerate(kernel_mul_list):
    #     print('kernel_mul: ' + str(kmul))
    #     loss_list.append([])
    #     for knum in kernel_num_list:
    #         print(' kernel_num: ' + str(knum))
    #         loss = MMD_loss(kernel_mul=kmul, kernel_num=knum)
    #         # print(loss.forward(full, full))
    #         # print(loss.forward(fast, fast))
    #         thisloss = loss.forward(full, fast).item()
    #         print('  ' + str(thisloss))
    #         loss_list[ikmul].append(thisloss)
    #         plt.text(kmul, knum, '{:0.5f}'.format(thisloss), ha='center', va='center')
    #
    # # bins = np.histogram(np.hstack(([x[0] for x in full_list], [x[0] for x in fast_list])), bins=100)[1]
    # # plt.hist([x[0] for x in full_list], bins, label='full', alpha=0.5)
    # # plt.hist([x[0] for x in fast_list], bins, label='fast', alpha=0.5)
    # # plt.legend()
    # # plt.show()
    #
    # plt.pcolormesh(kernel_mul_list, kernel_num_list, np.transpose(loss_list), shading='auto', vmin=np.min(loss_list), vmax=np.max(loss_list))
    # plt.title('MMD for 256 FastSim/FullSim jets defined by Response and Eta')
    # plt.xlabel('kernel_mul')
    # plt.ylabel('kernel_num')
    # plt.show()


def testMMDbandwidth():

    samplesize = 3
    test_samples = 1

    sigma_test = np.linspace(0.35, 2.5, test_samples)
    loc_test = np.linspace(-1, 1, test_samples)

    sigma_mmd = np.arange(0.5, 10, 0.5)

    bins = np.linspace(-3, 3, 30)

    reference = torch.tensor(np.random.normal(loc=0, scale=1, size=(samplesize, 1)))

    for i in range(test_samples):

        print(i+1, 'of', test_samples)

        f, axes = plt.subplots(2, 2, figsize=(16, 9), gridspec_kw={'width_ratios': [1, 1]})


        test = torch.tensor(np.random.normal(loc=loc_test[i], scale=1, size=(samplesize, 1)))

        loss_list = []
        for sgm in sigma_mmd:
            lossfunc = MMD_loss(kernel_mul=2, kernel_num=1, fix_sigma=sgm)
            lossval = lossfunc.forward(test, reference).item()
            loss_list.append(lossval)

        axes[0][0].hist(np.reshape(reference, (1, samplesize)), bins, label=r'reference $\mu=0, \sigma=1$', color='green', alpha=0.5)
        axes[0][0].hist(np.reshape(test, (1, samplesize)), bins, label=r'test $\mu=' + str(loc_test[i]) + ', \sigma=1$', color='red', alpha=1., histtype='step', linewidth=2)
        # axes[0][0].legend(loc='upper right')

        axes[0][1].bar(sigma_mmd, loss_list, width=0.1)
        axes[0][1].set_ylim(0, 0.2)
        axes[0][1].set_xlabel('bandwidth')
        axes[0][1].set_ylabel('MMD')


        test = torch.tensor(np.random.normal(loc=0, scale=sigma_test[i], size=(samplesize, 1)))

        loss_list = []
        for sgm in sigma_mmd:
            lossfunc = MMD_loss(kernel_mul=2, kernel_num=1, fix_sigma=sgm)
            lossval = lossfunc.forward(test, reference).item()
            loss_list.append(lossval)

        axes[1][0].hist(np.reshape(reference, (1, samplesize)), bins, label=r'reference $\mu=0, \sigma=1$', color='green', alpha=0.5)
        axes[1][0].hist(np.reshape(test, (1, samplesize)), bins, label=r'test $\mu=0, \sigma=' + str(sigma_test[i]) + '$', color='red', alpha=1., histtype='step', linewidth=2)
        # axes[1][0].legend(loc='upper right')

        axes[1][1].bar(sigma_mmd, loss_list, width=0.1)
        axes[1][1].set_ylim(0, 0.2)
        axes[1][1].set_xlabel('bandwidth')
        axes[1][1].set_ylabel('MMD')


        f.savefig('/afs/desy.de/user/w/wolfmor/Plots/Refinement/Training/Regression/MMD/MMDbandwidthtest_' + str(i).zfill(3) + '.png')

        # Clear the current axes.
        plt.cla()
        # Clear the current figure.
        plt.clf()
        # Closes all the figure windows.
        plt.close('all')


def testMMDparam():
    
    import numpy as np

    batchsize = 1000
    dims = 2

    param_num = 1
    # param_eps = 1


    mmd = MMD_loss(fix_sigma=1.)
    # mmd = MMD_loss()

    muA = 0.5
    muB = 1.
    muC = -0.5
    muD = -1.

    populationfactor = 3

    # sample 1
    A = torch.randn(batchsize, dims) + muA  # param=0
    B = torch.randn(batchsize*populationfactor, dims) + muB  # param=eps
    AB = torch.cat((A, B), dim=0)

    # sample 2
    C = torch.randn(batchsize, dims) + muC  # param=0
    D = torch.randn(batchsize*populationfactor, dims) + muD  # param=eps
    CD = torch.cat((C, D), dim=0)

    # print(A)
    # print(B)
    # print(C)
    # print(D)
    #
    # print(AB)
    # print(CD)

    mmdAC = mmd(A, C)
    mmdBD = mmd(B, D)
    mmdSum = mmdAC + mmdBD

    mmdABCD = mmd(AB, CD)

    print('')
    print(mmdAC)
    print(mmdBD)
    print(mmdSum)
    print(mmdABCD)

    # mmdACx0 = mmd(A, C, parameters=torch.full((batchsize, param_num), 0))
    # mmdBDx0 = mmd(B, D, parameters=torch.full((batchsize, param_num), 0))
    # mmdSumx0 = mmdACx0 + mmdBDx0
    #
    # print('')
    # print(mmdACx0)
    # print(mmdBDx0)
    # print(mmdSumx0)
    #
    # mmdACx1 = mmd(A, C, parameters=torch.full((batchsize, param_num), 1))
    # mmdBDx1 = mmd(B, D, parameters=torch.full((batchsize, param_num), 1))
    # mmdSumx1 = mmdACx1 + mmdBDx1
    #
    # print('')
    # print(mmdACx1)
    # print(mmdBDx1)
    # print(mmdSumx1)
    
    
    mmds = []
    param_eps_list = np.logspace(-3., 3.)

    for param_eps in param_eps_list:

        print('')
        print(param_eps)

        # Ap = torch.cat((A, torch.full((batchsize, param_num), 0)), dim=1)
        # Bp = torch.cat((B, torch.full((batchsize, param_num), param_eps)), dim=1)
        # ABp = torch.cat((Ap, Bp), dim=0)
        #
        # Cp = torch.cat((C, torch.full((batchsize, param_num), 0)), dim=1)
        # Dp = torch.cat((D, torch.full((batchsize, param_num), param_eps)), dim=1)
        # CDp = torch.cat((Cp, Dp), dim=0)

        p = torch.cat((torch.full((batchsize, param_num), 0), torch.full((batchsize*populationfactor, param_num), param_eps)), dim=0)

        # print(Ap)
        # print(Bp)
        # print(Cp)
        # print(Dp)
        #
        # print(ABp)
        # print(CDp)
        #
        # print(p)

        thismmd = mmd(AB, CD, parameters=p)

        mmds.append(thismmd)
        print(thismmd)
        # print(mmd(ABp, CDp))

    torch.Tensor.ndim = property(lambda self: len(self.shape))  # to plot pytorch tensors with pyplot

    f, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].plot(A[:, 0], A[:, 1], 'bo', label='$A^{0}\ (\mu=' + str(muA) + ',\ z=0, n_{0}=' + str(batchsize) + ')$')
    axes[0].plot(B[:, 0], B[:, 1], 'bX', label='$A^{1}\ (\mu=' + str(muB) + ',\ z=\epsilon, n_{1}=' + str(batchsize*populationfactor) + ')$')
    axes[0].plot(C[:, 0], C[:, 1], 'ro', label='$B^{0}\ (\mu=' + str(muC) + ',\ z=0, n_{0}=' + str(batchsize) + ')$')
    axes[0].plot(D[:, 0], D[:, 1], 'rX', label='$B^{1}\ (\mu=' + str(muD) + ',\ z=\epsilon, n_{1}=' + str(batchsize*populationfactor) + ')$')

    axes[0].legend()

    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')


    axes[1].axhline(mmdAC, ls='dashed', label='$MMD_{xy}(A^{0}, B^{0})$')
    axes[1].axhline(mmdBD, ls='dotted', label='$MMD_{xy}(A^{1}, B^{1})$')
    axes[1].axhline(((1./(1. + populationfactor))**2. * mmdAC + (populationfactor/(1. + populationfactor))**2. * mmdBD), ls='dashdot', label=r'$(\frac{n_{0}}{n_{0}+n_{1}})^{2}\times MMD_{xy}(A^{0}, B^{0}) + (\frac{n_{1}}{n_{0}+n_{1}})^{2}\times MMD_{xy}(A^{1}, B^{1})$')
    axes[1].axhline(mmdABCD, ls='solid', label='$MMD_{xy}(A^{0}+A^{1}, B^{0}+B^{1})$')

    # axes[1].axhline(0.5 * mmdSum, ls='dashed', label='$(MMD_{xy}(A^{0}, B^{0}) + MMD_{xy}(A^{1}, B^{1})) / 2$')
    # axes[1].axhline(0.25 * mmdSum, ls='dotted', label='$(MMD_{xy}(A^{0}, B^{0}) + MMD_{xy}(A^{1}, B^{1})) / 4$')

    axes[1].plot(param_eps_list, mmds, marker='x', label='$MMD_{xyz}(A^{0}+A^{1}, B^{0}+B^{1})$')

    axes[1].legend()
    axes[1].set_xscale('log')

    axes[1].set_xlabel('$\epsilon$')
    axes[1].set_ylabel('MMD')

    plt.tight_layout()
    f.savefig('/afs/desy.de/user/w/wolfmor/Plots/Refinement/Training/Regression/MMD/MMDparamtest.png')


def testMMDmask():

    X = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]])
    Y = torch.tensor([[0, 1], [1, 0], [2, 0], [0, 2]])



if __name__ == '__main__':

    import ROOT

    # in_path = '/nfs/dust/cms/user/wolfmor/Refinement/littletree_CMSSW_12_2_3_T1tttt_step3_inNANOAODSIM_coffea.root'
    in_path = '/nfs/dust/cms/user/wolfmor/Refinement/littletree_CMSSW_12_6_0_TTbar_step2_SIM_RECOBEFMIX_DIGI_L1_DIGI2RAW_L1Reco_RECO_PAT_NANO_coffea_new.root'
    in_path = '/nfs/dust/cms/user/wolfmor/Refinement/TrainingOutput/output_regression_20230218_1.root'
    fin = ROOT.TFile.Open(in_path)
    tree = fin.Get('tJet')

    # tempfile = ROOT.TFile.Open('tempfile_DeepFlavsPlusParams_TTbar_12_6_RecHadFlav0.root', 'recreate')
    # tree = tree.CopyTree('RecJet_hadronFlavour_FastSim==0')
    # tempfile.Write()

    testMMD(tree)

    # testMMDmask()
    # testMMDparam()

    pass

    # mmd = MMD_loss(kernel_mul=2.0, kernel_num=1, fix_sigma=1.)
    # # mmd = MMD_loss(kernel_mul=2.0, kernel_num=3)
    #
    # X = torch.tensor([[0, 0], [1, 1]])
    # Y = torch.tensor([[0, 1], [1, 0]])
    # Z = torch.tensor([[0.1, 0], [0.9, 1]])
    #
    # print(mmd(X, Y))
    # print(mmd(X, Z))


