"""
#--------------------------------------------------------------------------------------#
 
Test to compare the timings(on GPU) of operations, Grouped Convolution and native convolution using For loop and F.conv2d. 

#---------------------------------------------------------------------------------------#
"""

import timeit


setup = '''

import torch
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

img = torch.rand((64,256,24,24), device=device)  # standard output of net1
filters = torch.rand((64,100,3,3), device=device)   # standard output of net2 

'''

group_conv = '''
	
batch_size, c_in, h1, w1 = img.shape
_, c_out, h2, w2 = filters.shape

filters_ = filters[:, :, None, :, :]
filters_ = filters_.repeat(1, 1, c_in, 1, 1)

out =  F.conv2d(
    input=img.view(1, batch_size * c_in, h1, w1),
    weight=filters_.view(batch_size * c_out, c_in, h2, w2),
    groups=batch_size)
'''

native_conv = '''

batch_size, c_in, h1, w1 = img.shape
_, c_out, h2, w2 = filters.shape

filters_ = filters[:, :, None, :, :]
filters_ = filters_.repeat(1, 1, c_in, 1, 1)

out = torch.zeros((batch_size, c_out, h1-h2+1, w1-w2+1))

for i in range(batch_size):
    i_filter = filters_[i]
    i_input = img[i][None, :, :, :]
    out[i] = F.conv2d(i_input, i_filter)
'''

number = 1000
print(f'Avg time for native_conv is : {timeit.timeit(setup=setup, stmt=native_conv, number=number)/number} seconds')
print(f'Avg time for group_conv is : {timeit.timeit(setup=setup, stmt=group_conv, number=number)/number} seconds')


"""
#--------------Output on Colab's GPU----------------------------#

	Avg time for native_conv is : 0.040258811977000736 seconds
	Avg time for group_conv is : 0.02103576725200037 seconds
	
#----------------------------------------------------------------#
"""