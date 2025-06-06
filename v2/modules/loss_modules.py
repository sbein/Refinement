import torch
from torch import nn

class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, unbiased=False):
        """
        Args:
            kernel_type (str): Type of kernel to use. Currently only 'rbf' is supported.
            kernel_mul (float or list): Factor(s) to multiply the median distance for calculating
                                       the RBF kernel bandwidth (sigma).
                                       If it's a list, multi-kernel MMD is used.
            kernel_num (int): Number of kernels to use for multi-kernel MMD.
                             If fix_sigma is not provided and kernel_mul is a float,
                             kernel_num sigmas will be created around kernel_mul.
            fix_sigma (float or list): If not None, this sigma/list of sigmas will be used directly.
                                      In this case, kernel_mul and kernel_num are ignored.
            unbiased (bool): If True, uses the unbiased MMD estimator, if False uses the biased one.
        """
        super(MMDLoss, self).__init__()
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma
        self.unbiased = unbiased

        if self.kernel_type not in ['rbf']:
            raise ValueError(f"Unsupported kernel type: {kernel_type}. Only 'rbf' is supported.")

    def _guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples_source = source.size(0)
        n_samples_target = target.size(0)

        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
        total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
        
        X_sq = torch.sum(source**2, dim=1, keepdim=True) 
        Y_sq = torch.sum(target**2, dim=1, keepdim=True) 
        XY = torch.matmul(source, target.t())            

        dist_ss_sq = X_sq + X_sq.t() - 2 * torch.matmul(source, source.t()) 
        dist_tt_sq = Y_sq + Y_sq.t() - 2 * torch.matmul(target, target.t()) 
        dist_st_sq = X_sq + Y_sq.t() - 2 * XY                               

        if fix_sigma is not None:
            if isinstance(fix_sigma, (float, int)):
                sigma_list = [float(fix_sigma)]
            elif isinstance(fix_sigma, list):
                sigma_list = [float(s) for s in fix_sigma]
            else:
                raise ValueError("fix_sigma must be a float or a list of floats.")
        else:
            all_dist_sq = torch.cat([
                dist_ss_sq[torch.triu(torch.ones_like(dist_ss_sq), diagonal=1) == 1], 
                dist_tt_sq[torch.triu(torch.ones_like(dist_tt_sq), diagonal=1) == 1],
                dist_st_sq.view(-1)
            ])
            
            median_dist_sq = torch.median(all_dist_sq[all_dist_sq > 1e-9]) + 1e-9 
            if torch.isnan(median_dist_sq) or median_dist_sq <= 1e-9: 
                median_dist_sq = torch.tensor(1.0, device=source.device) 

            if isinstance(kernel_mul, (float, int)):
                sigma_list = [median_dist_sq.item() * kernel_mul] 
                                                                  
                if kernel_num > 1:
                    base_sigma_sq_x2 = sigma_list[0] 
                    
                    factors = [0.25, 0.5, 1.0, 2.0, 4.0] 
                    if self.kernel_num == 1: factors = [1.0]
                    elif self.kernel_num == 3: factors = [0.5, 1.0, 2.0]
                    
                    
                    if self.kernel_num not in [1,3,5]:
                        print(f"WARNING: Using default factors for kernel_num={self.kernel_num} (similar to kernel_num=5).")

                    sigma_list = [base_sigma_sq_x2 * f for f in factors[:self.kernel_num]]

            elif isinstance(kernel_mul, list):
                sigma_list = [median_dist_sq.item() * mul for mul in kernel_mul]
            else:
                raise ValueError("kernel_mul must be a float or a list of floats.")
        
        K_ss_total = torch.zeros_like(dist_ss_sq, device=source.device)
        K_tt_total = torch.zeros_like(dist_tt_sq, device=source.device)
        K_st_total = torch.zeros_like(dist_st_sq, device=source.device)

        for sigma_val_sq_x2 in sigma_list:
            if sigma_val_sq_x2 <= 1e-9: 
                sigma_val_sq_x2 = 1e-9
            K_ss_total += torch.exp(-dist_ss_sq / sigma_val_sq_x2)
            K_tt_total += torch.exp(-dist_tt_sq / sigma_val_sq_x2)
            K_st_total += torch.exp(-dist_st_sq / sigma_val_sq_x2)
            
        return K_ss_total / len(sigma_list), \
               K_tt_total / len(sigma_list), \
               K_st_total / len(sigma_list)

    def forward(self, source, target):
        if source.size(1) != target.size(1):
            raise ValueError(f"Source and target feature dimensions must match: {source.size(1)} != {target.size(1)}")
        
        m = source.size(0)
        n = target.size(0)

        if self.kernel_type == 'rbf':
            K_ss, K_tt, K_st = self._guassian_kernel(source, target, self.kernel_mul, self.kernel_num, self.fix_sigma)
        else:
            #TODO: add new kernels
            raise NotImplementedError

        if self.unbiased:

            if m < 2: 
                term1 = torch.tensor(0.0, device=source.device)
            else:
                term1 = (K_ss.sum() - K_ss.diag().sum()) / (m * (m - 1))
            
            if n < 2:
                term2 = torch.tensor(0.0, device=source.device)
            else:
                term2 = (K_tt.sum() - K_tt.diag().sum()) / (n * (n - 1))
            
            term3 = 2 * K_st.mean() 
            
            loss = term1 + term2 - term3
        else:
            
            term1 = K_ss.mean() 
            term2 = K_tt.mean() 
            term3 = 2 * K_st.mean() 
            
            loss = term1 + term2 - term3
            
        return loss

losses = {
    'l1': nn.L1Loss,
    'l2': nn.MSELoss,
    'mse': nn.MSELoss,
    'huber': nn.HuberLoss,
    'bce': nn.BCELoss,
    'bce_logits': nn.BCEWithLogitsLoss,
    'cross_entropy': nn.CrossEntropyLoss,
    'smooth_l1': nn.SmoothL1Loss,
    'mmd': MMDLoss
}