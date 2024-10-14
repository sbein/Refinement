import sys

import torch
from torch import nn


class MMD(nn.Module):
    """Implementation of the empirical estimate of the maximum mean discrepancy

    adapted from https://github.com/ZongxianLee/MMD_Loss.Pytorch/blob/master/mmd_loss.py
    see https://en.wikipedia.org/wiki/Kernel_embedding_of_distributions#Measuring_distance_between_distributions

    the MMD values for n="kernel_num" Gaussian kernels with different bandwidths are added
    "bandwidth" is the central bandwidth (equivalent to 2*sigma^2 for standard Gaussian)
    the bandwidths range from "bandwidth" / ("kernel_mul" * "kernel_num" // 2) to "bandwidth" * ("kernel_mul" * "kernel_num" // 2)
    "bandwidth" can be fixed or is calculated from the L2-distance

    """
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None, one_sided_bandwidth=False, exclude_diagonals=False, weighted=False,
                 calculate_fix_sigma_with_target_only=False, calculate_fix_sigma_for_each_dimension_with_target_only=False):

        super(MMD, self).__init__()

        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma
        self.one_sided_bandwidth = one_sided_bandwidth
        self.exclude_diagonals = exclude_diagonals
        self.weighted = weighted

        self.calculate_fix_sigma_with_target_only = calculate_fix_sigma_with_target_only  # fix_sigma will be overwritten
        self.fix_sigma_target_only = None

        self.calculate_fix_sigma_for_each_dimension_with_target_only = calculate_fix_sigma_for_each_dimension_with_target_only  # fix_sigma will be overwritten
        self.fix_sigma_target_only_by_dimension = None

        return

    @staticmethod
    def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None, one_sided_bandwidth=False, l2dist_out=None):

        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size()[0]), int(total.size()[0]), int(total.size()[1]))
        total1 = total.unsqueeze(1).expand(int(total.size()[0]), int(total.size()[0]), int(total.size()[1]))

        if type(fix_sigma) == list or (torch.is_tensor(fix_sigma) and fix_sigma.size()[0] > 1):
            L2_distance = ((total0-total1)**2)
        else:
            L2_distance = ((total0-total1)**2).sum(2)

        if l2dist_out is not None:
            globals()[l2dist_out] = (torch.sum(((total0-total1)**2).sum(2).data) / (n_samples**2-n_samples)).item()

        if fix_sigma is None:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        else:
            bandwidth = fix_sigma.detach().clone() if torch.is_tensor(fix_sigma) else torch.tensor(fix_sigma, device=source.device)

        if one_sided_bandwidth:
            bandwidth /= kernel_mul ** (kernel_num - 1)
        else:
            bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

        if type(fix_sigma) == list or (torch.is_tensor(fix_sigma) and fix_sigma.size()[0] > 1):
            kernel_val = [torch.exp(-(L2_distance / bandwidth_temp).sum(2)) for bandwidth_temp in bandwidth_list]
        else:
            kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

        # some printouts that might be useful
        # print('\nMMD:')
        # print(L2_distance)
        # print(L2_distance / bandwidth_list[0])
        # print(torch.exp(-(L2_distance / bandwidth_list[0])))
        # print(torch.exp(-(L2_distance / bandwidth_list[0]).sum(2)))
        # print('L2 distance')
        # print(torch.sum(L2_distance.data) / (n_samples**2-n_samples))
        # print('bandwidths')
        # print(bandwidth_list)
        # print('mmd values')
        # print([torch.mean(k[:int(source.size()[0]), :int(source.size()[0])] + k[int(source.size()[0]):, int(source.size()[0]):] - k[:int(source.size()[0]), int(source.size()[0]):] - k[int(source.size()[0]):, :int(source.size()[0])]) for k in kernel_val])

        return sum(kernel_val)

    def forward(self, source, target, parameters=None, mask=None, l2dist_out=None):

        nvariables = source.size()[1]

        if parameters is not None:
            source = torch.cat((source, parameters), dim=1)
            target = torch.cat((target, parameters), dim=1)

        if mask is not None:
            source = source[mask]
            target = target[mask]


        if self.calculate_fix_sigma_with_target_only:  # TODO: remove?

            if self.fix_sigma_target_only is None:

                n_samples = int(target.size()[0])

                total = target

                total0 = total.unsqueeze(0).expand(int(total.size()[0]), int(total.size()[0]), int(total.size()[1]))
                total1 = total.unsqueeze(1).expand(int(total.size()[0]), int(total.size()[0]), int(total.size()[1]))
                L2_distance = ((total0-total1)**2).sum(2)

                self.fix_sigma_target_only = torch.sum(L2_distance.data) / (n_samples**2-n_samples)

                print('\ncalculated BW to be:')
                print(self.fix_sigma_target_only)

            sigma = self.fix_sigma_target_only

        elif self.calculate_fix_sigma_for_each_dimension_with_target_only:

            if self.fix_sigma_target_only_by_dimension is None:

                # TODO: use target and source?

                total = torch.cat([source, target], dim=0)

                total0 = total.unsqueeze(0).expand(int(total.size()[0]), int(total.size()[0]), int(total.size()[1]))
                total1 = total.unsqueeze(1).expand(int(total.size()[0]), int(total.size()[0]), int(total.size()[1]))
                L2_distances = (total0-total1)**2

                if type(self.calculate_fix_sigma_for_each_dimension_with_target_only) == str:
                    if self.calculate_fix_sigma_for_each_dimension_with_target_only == 'median':
                        self.fix_sigma_target_only_by_dimension = torch.tensor([L2_distances[:, :, i].median() for i in range(total.size()[1])], device=total.device)
                    elif self.calculate_fix_sigma_for_each_dimension_with_target_only == 'mean':
                        self.fix_sigma_target_only_by_dimension = torch.tensor([L2_distances[:, :, i].mean() for i in range(total.size()[1])], device=total.device)
                    else:
                        raise NotImplementedError('cannot understand how to calculate MMD bandwidths: ' + self.fix_sigma_target_only_by_dimension)
                else:
                    self.fix_sigma_target_only_by_dimension = torch.tensor([L2_distances[:, :, i].median() for i in range(total.size()[1])], device=total.device)

                print('\ncalculated BWs by dimension to be:')
                print(self.fix_sigma_target_only_by_dimension)

                for ibw, bw in enumerate(self.fix_sigma_target_only_by_dimension):
                    if bw <= 0: raise Exception('bandwidth is <= 0 for variable number' + str(ibw))

            sigma = self.fix_sigma_target_only_by_dimension

        else:

            sigma = self.fix_sigma


        batch_size = int(source.size()[0])
        kernels = MMD.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=sigma,
                                      one_sided_bandwidth=self.one_sided_bandwidth, l2dist_out=l2dist_out)

        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]

        if self.exclude_diagonals:

            XX = XX * (1 - torch.eye(batch_size, device=XX.device))
            YY = YY * (1 - torch.eye(batch_size, device=YY.device))

        if self.weighted:

            with torch.no_grad():

                bn = torch.nn.BatchNorm1d(nvariables, affine=False, track_running_stats=False, device=source.device)

                source = bn(source[:, :nvariables])
                target = bn(target[:, :nvariables])

                source = torch.abs(source)
                target = torch.abs(target)

                source = torch.clamp(source, min=1)
                target = torch.clamp(target, min=1)

                p = 1.
                source = torch.pow(source, p).mean(dim=1)
                target = torch.pow(target, p).mean(dim=1)
                # source, _ = torch.max(torch.abs(source), dim=1)
                # target, _ = torch.max(torch.abs(target), dim=1)

                weightXX = torch.sqrt(source.unsqueeze(0) * source.unsqueeze(1))
                weightYY = torch.sqrt(target.unsqueeze(0) * target.unsqueeze(1))
                weightXY = torch.sqrt(source.unsqueeze(0) * target.unsqueeze(1))
                weightYX = torch.sqrt(target.unsqueeze(0) * source.unsqueeze(1))

            XX = XX * weightXX
            YY = YY * weightYY
            XY = XY * weightXY
            YX = YX * weightYX

        if batch_size == 0:  # if nothing survives the mask

            loss = torch.tensor(0.)

        else:

            if self.exclude_diagonals:

                loss = XX.sum() / (batch_size * batch_size - batch_size) \
                    + YY.sum() / (batch_size * batch_size - batch_size) \
                    - XY.sum() / (batch_size * batch_size) \
                    - YX.sum() / (batch_size * batch_size)

            else:

                loss = torch.mean(XX + YY - XY - YX)

        return loss


if __name__ == '__main__':

    pass
