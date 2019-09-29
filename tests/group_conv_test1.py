#--------------------------------------------------------------------------------------#
# 
# Test to make sure the output of Grouped Convolution is same 
# 	as the one using native for loops and F.conv2d.
#
#---------------------------------------------------------------------------------------#


import torch
import torch.nn.functional as F


def grouped_conv(img, filters):

	batch_size, c_in, h1, w1 = img.shape
	_, c_out, h2, w2 = filters.shape

	filters = filters[:, :, None, :, :]
	filters = filters.repeat(1, 1, c_in, 1, 1)

	return F.conv2d(
	    input=img.view(1, batch_size * c_in, h1, w1),
	    weight=filters.view(batch_size * c_out, c_in, h2, w2),
	    groups=batch_size)


def native_conv(img, filters):
	
	batch_size, c_in, h1, w1 = img.shape
	_, c_out, h2, w2 = filters.shape

	filters = filters[:, :, None, :, :]
	filters = filters.repeat(1, 1, c_in, 1, 1)

	out = torch.zeros((batch_size, c_out, h1-h2+1, w1-w2+1))

	for i in range(batch_size):
	    i_filter = filters[i]
	    i_input = img[i][None, :, :, :]
	    out[i] = F.conv2d(i_input, i_filter)

	return out

if __name__ == "__main__":

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	img = torch.rand((64,256,24,24), device=device)  # standard output of net1
	filters = torch.rand((64,100,3,3), device=device)   # standard output of net2

	out1 = grouped_conv(img, filters)
	out2 = native_conv(img, filters)

	print(torch.allclose(out1.view_as(out2), out2))


