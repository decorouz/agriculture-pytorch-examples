# Identify specific kind of crop in a farm plot.
# Suppose we’re analyzing satellite data to monitor a farm with multiple plots,
# each growing different crops.
# We use a tensor to represent the farm plots, where each number in the tensor corresponds to a specific crop type:

# 0: Empty (no crop)
# 1: Maize
# 2: Cowpea
# 3: Soybeans

import torch

# Farm plot data (tensor), where each number represents a crop type
farm_plot = torch.tensor([[1, 2, 3, 0], [3, 1, 0, 2], [0, 2, 2, 1], [3, 0, 1, 3]])

# Let find all the location of all the plots growing maize (represented by '1' in our tensor)

maize_plots = torch.argwhere(farm_plot == 1)
print(maize_plots)
