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
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None, one_sided_bandwidth=False,
                 calculate_sigma_without_parameters=False, calculate_fix_sigma_with_target_only=False, calculate_fix_sigma_for_each_dimension_with_target_only=False):

        super(MMD, self).__init__()

        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma
        self.one_sided_bandwidth = one_sided_bandwidth

        self.calculate_sigma_without_parameters = calculate_sigma_without_parameters  # fix_sigma will be overwritten

        self.calculate_fix_sigma_with_target_only = calculate_fix_sigma_with_target_only  # fix_sigma will be overwritten
        self.fix_sigma_target_only = None

        self.calculate_fix_sigma_for_each_dimension_with_target_only = calculate_fix_sigma_for_each_dimension_with_target_only  # fix_sigma will be overwritten
        self.fix_sigma_target_only_by_dimension = None

        return

    @staticmethod
    def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None, one_sided_bandwidth=False, bandwidth_by_dimension=False, l2dist_out=None):

        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size()[0]), int(total.size()[0]), int(total.size()[1]))
        total1 = total.unsqueeze(1).expand(int(total.size()[0]), int(total.size()[0]), int(total.size()[1]))

        if bandwidth_by_dimension:
            L2_distance = ((total0-total1)**2)
        else:
            L2_distance = ((total0-total1)**2).sum(2)

        if l2dist_out is not None:
            globals()[l2dist_out] = (torch.sum(((total0-total1)**2).sum(2).data) / (n_samples**2-n_samples)).item()

        if fix_sigma is None:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        else:
            bandwidth = fix_sigma if torch.is_tensor(fix_sigma) else torch.tensor(fix_sigma)
        if one_sided_bandwidth:
            bandwidth /= kernel_mul ** (kernel_num - 1)
        else:
            bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

        if bandwidth_by_dimension:
            kernel_val = [torch.exp(-(L2_distance / bandwidth_temp).sum(2)) for bandwidth_temp in bandwidth_list]
        else:
            kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

        # some printouts that might be useful
        # print('\nMMD:')
        # print('L2 distance')
        # print(torch.sum(L2_distance.data) / (n_samples**2-n_samples))
        # print('bandwidths')
        # print(bandwidth_list)
        # print('mmd values')
        # print([torch.mean(k[:int(source.size()[0]), :int(source.size()[0])] + k[int(source.size()[0]):, int(source.size()[0]):] - k[:int(source.size()[0]), int(source.size()[0]):] - k[int(source.size()[0]):, :int(source.size()[0])]) for k in kernel_val])

        return sum(kernel_val)

    def forward(self, source, target, parameters=None, mask=None, l2dist_out=None):

        if self.calculate_fix_sigma_with_target_only:

            if self.fix_sigma_target_only is None:

                n_samples = int(target.size()[0])

                if parameters is not None:
                    total = torch.cat((target, parameters), dim=1)
                else:
                    total = target

                if mask is not None:
                    total = total[mask]

                total0 = total.unsqueeze(0).expand(int(total.size()[0]), int(total.size()[0]), int(total.size()[1]))
                total1 = total.unsqueeze(1).expand(int(total.size()[0]), int(total.size()[0]), int(total.size()[1]))
                L2_distance = ((total0-total1)**2).sum(2)

                self.fix_sigma_target_only = torch.sum(L2_distance.data) / (n_samples**2-n_samples)

            sigma = self.fix_sigma_target_only

        elif self.calculate_fix_sigma_for_each_dimension_with_target_only:

            if self.fix_sigma_target_only_by_dimension is None:

                n_samples = int(target.size()[0])

                if parameters is not None:
                    total = torch.cat((target, parameters), dim=1)
                else:
                    total = target

                if mask is not None:
                    total = total[mask]

                total0 = total.unsqueeze(0).expand(int(total.size()[0]), int(total.size()[0]), int(total.size()[1]))
                total1 = total.unsqueeze(1).expand(int(total.size()[0]), int(total.size()[0]), int(total.size()[1]))
                L2_distance = ((total0-total1)**2).sum((0, 1))

                self.fix_sigma_target_only_by_dimension = L2_distance.data / (n_samples**2-n_samples)

            sigma = self.fix_sigma_target_only_by_dimension

        elif self.calculate_sigma_without_parameters:

            n_samples = int(source.size()[0])+int(target.size()[0])

            if mask is not None:
                total = torch.cat([source[mask], target[mask]], dim=0)
            else:
                total = torch.cat([source, target], dim=0)

            total0 = total.unsqueeze(0).expand(int(total.size()[0]), int(total.size()[0]), int(total.size()[1]))
            total1 = total.unsqueeze(1).expand(int(total.size()[0]), int(total.size()[0]), int(total.size()[1]))
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

        batch_size = int(source.size()[0])
        kernels = MMD.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=sigma,
                                      one_sided_bandwidth=self.one_sided_bandwidth, bandwidth_by_dimension=self.calculate_fix_sigma_for_each_dimension_with_target_only,
                                      l2dist_out=l2dist_out)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        if batch_size == 0:
            # if nothing survives the mask
            loss = torch.tensor(0.)
        else:
            loss = torch.mean(XX + YY - XY - YX)

        return loss


if __name__ == '__main__':

    pass
