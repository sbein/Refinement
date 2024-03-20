import torch
from torch import nn
import math

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
            L1 = total0-total1
        else:
            L2_distance = ((total0-total1)**2).sum(2)
            
        if torch.isnan(L2_distance).any():
            print("Situation with L2_distance")
            # Optional: print some values to inspect
            print("Sample differences:", (total0-total1))

            print('This gave a NaN', torch.isnan(L2_distance).any(),torch.isnan(total0).any(),torch.isnan(total1).any(),torch.isnan(L1).any())
            diff = total0 - total1
            nan_indices = torch.isnan(diff).nonzero()
            print("Indices of NaN values in difference:", nan_indices)
            for idx in nan_indices:
                # Unpack the indices
                i, j, k = idx.tolist()
                
                # Print the problematic values
                print(f"Inspecting NaN at index [{i}, {j}, {k}]:")
                print(f"Value in total0: {total0[i, j, k].item()}")
                print(f"Value in total1: {total1[i, j, k].item()}")
                print("-" * 40)              
            exit(0)

        if l2dist_out is not None:
            globals()[l2dist_out] = (torch.sum(((total0-total1)**2).sum(2).data) / (n_samples**2-n_samples)).item()

        if fix_sigma is None:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        else:
            bandwidth = fix_sigma.detach().clone() if torch.is_tensor(fix_sigma) else torch.tensor(fix_sigma)
        if one_sided_bandwidth:
            bandwidth /= kernel_mul ** (kernel_num - 1)
        else:
            bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        
        if torch.isnan(bandwidth).any():
            print("NaN detected in bandwidth calculation")
    
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

        for i, kv in enumerate(kernel_val):
            if torch.isnan(kv).any():
                print(f"NaN detected in kernel_val at index {i}")
                
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

                print('\ncalculated BW to be:')
                print(self.fix_sigma_target_only)

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
                if torch.isnan(total0).any() or torch.isnan(total1).any():
                    print("NaN detected in total0 or total1 before computing L2_distance")                
                L2_distance = ((total0-total1)**2).sum((0, 1))
                if torch.isnan(L2_distance).any():
                    print("NaN detected in L2_distance B")
                    # Optional: print some values to inspect
                    print("Sample values in total0:", total0[0][0][:10])
                    print("Sample values in total1:", total1[0][0][:10])
                    print("Sample differences:", (total0-total1)[0][0][:10])

                self.fix_sigma_target_only_by_dimension = L2_distance.data / (n_samples**2-n_samples)

                print('\ncalculated BWs by dimension to be:')
                print(self.fix_sigma_target_only_by_dimension)

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
            if torch.isnan(L2_distance).any():
                print("NaN detected in L2_distance C")
                # Optional: print some values to inspect
                print("Sample values in total0:", total0[0][0][:10])
                print("Sample values in total1:", total1[0][0][:10])
                print("Sample differences:", (total0-total1)[0][0][:10])            

            sigma = torch.sum(L2_distance.data) / (n_samples**2-n_samples)

        else:

            sigma = self.fix_sigma
            
            
        if sigma is not None:
            if torch.is_tensor(sigma):
                if torch.isnan(sigma).any():
                    print("NaN detected in sigma tensor inside forward")
            else:
                if math.isnan(sigma):  # For non-tensor, retain the original check
                    print("NaN detected in sigma (non-tensor) inside forward")
                

        if parameters is not None:
            source = torch.cat((source, parameters), dim=1)
            target = torch.cat((target, parameters), dim=1)

        if mask is not None:
            source = source[mask]
            target = target[mask]

        batch_size = int(source.size()[0])
        if torch.isnan(source).any():
            print("NaN detected in source before gaussian_kernel")
        if torch.isnan(target).any():
            print("NaN detected in target before gaussian_kernel")
            
        kernels = MMD.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=sigma,
                                      one_sided_bandwidth=self.one_sided_bandwidth, bandwidth_by_dimension=self.calculate_fix_sigma_for_each_dimension_with_target_only,
                                      l2dist_out=l2dist_out)
    
        if torch.isnan(kernels).any():
            print("NaN detected in kernels output from gaussian_kernel")

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
