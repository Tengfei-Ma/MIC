import torch
from matplotlib import pyplot as plt

import torch.nn.functional as F

image = torch.rand(3, 512, 512)
attention_map = torch.rand(128, 128, 1)

plt.imshow(attention_map.detach().cpu().numpy(), cmap='jet', interpolation='nearest')
plt.colorbar()  # 添加颜色条
plt.show()
plt.close()

flatten_attention = attention_map.view(-1)
threshold_index = int(0.3 * flatten_attention.numel())
threshold_value, _ = torch.kthvalue(flatten_attention, threshold_index)
input_mask = (attention_map < threshold_value).float()
input_mask = input_mask.permute(2, 0, 1).unsqueeze(0)
input_mask = F.interpolate(input_mask, size=(512, 512), mode='bilinear', align_corners=False)
input_mask = input_mask.squeeze(0)
masked_img = image * input_mask
plt.imshow(masked_img.permute(1, 2, 0).numpy())
plt.axis('off')  # 不显示坐标轴
plt.show()
plt.close()
