import numpy as np

# generate random images for MNIST dataset
def rand_gen(n_samples, kernel=(1,1), shape=(28,28)):
    '''
    Produce a randomly-generated image
    --------------------------------------------------------
    Parameters
    n_samples:  number of samples to generate
    kernel:     parameters for the generative beta distribution
    shape:      specify shape of each sample to be generated
    --------------------------------------------------------
    Outputs
    samples:    ndarray of samples with shape (n_samples, *shape)
    '''
    kernel = np.array(kernel)
    
    # single kernel for whole set
    if len(kernel.shape) == 1 and len(kernel) == 2:
        return np.random.beta(kernel[0], kernel[1], size=(n_samples, *shape))

    # separate kernel for each pixel (last size two)
    if len(kernel.shape) == 1 + len(shape) and kernel.shape[-1] == 2:
        kernel = kernel.reshape((-1,2))
        gen = np.array([np.random.beta(a, b, size=(n_samples)) for a,b in kernel]).transpose()
        return gen.reshape(n_samples, *shape)

    assert False, 'invalid inputs'

# convert a grayscale image to a kernel for noisy image generation
def img_to_kernel(img, preservation, target='mode'):
    '''
    Produce a randomly-generated image
    --------------------------------------------------------
    Parameters
    img:          original image(s), values normalized to between 0 and 1
    preservation: how strongly to preserve original image (float > 0 if mode, >= 0 if mean)
    target:       whether to use 'mean' targeting or 'mode' targeting for kernel
    --------------------------------------------------------
    Outputs
    kernel:       generative kernel as ndarray, shape (*(img.shape), 2)
    '''
    img = np.array(img)
    flat = img.reshape(-1)

    kernel = None
    if target == 'mode':
        kernel = np.ones((len(flat), 2))
    elif target == 'mean':
        kernel = np.zeros((len(flat), 2))
    else:
        assert False, 'invalid target input'
    

    for i in range(len(flat)):
        kernel[i, 0] += preservation*flat[i]
        kernel[i, 1] += preservation*(1-flat[i])
    
    return kernel.reshape(*(img.shape), 2)

# unit test(s)
if __name__ == '__main__':

    import matplotlib.pyplot as plt

    n = 2

    # single kernel
    samples = rand_gen(n)
    assert len(samples) == n
    plt.figure()
    plt.title('Uniform Kernel')
    plt.imshow(samples[0], cmap='gray')
    # plt.show(block=True)

    # pixel-wise kernel
    st = 3
    k = np.array([
        [[st,1], [st,1], [st,1], [st,1], [st,1], [1,st], [st,1], [st,1], [st,1], [st,1], [st,1]],
        [[st,1], [st,1], [1,st], [st,1], [st,1], [1,st], [st,1], [1,st], [st,1], [st,1], [st,1]],
        [[st,1], [st,1], [1,st], [st,1], [st,1], [1,st], [st,1], [1,st], [st,1], [1,st], [st,1]],
        [[st,1], [st,1], [1,st], [st,1], [st,1], [1,st], [st,1], [1,st], [st,1], [1,st], [st,1]],
        [[st,1], [st,1], [1,st], [st,1], [st,1], [1,st], [st,1], [1,st], [st,1], [1,st], [st,1]],
        [[1,st], [1,st], [1,st], [1,st], [1,st], [1,st], [1,st], [1,st], [1,st], [1,st], [1,st]],
        [[st,1], [1,st], [st,1], [1,st], [st,1], [1,st], [st,1], [1,st], [st,1], [st,1], [st,1]],
        [[st,1], [1,st], [st,1], [1,st], [st,1], [1,st], [st,1], [1,st], [st,1], [st,1], [st,1]],
        [[st,1], [1,st], [st,1], [1,st], [st,1], [1,st], [st,1], [1,st], [1,st], [1,st], [st,1]],
        [[st,1], [1,st], [st,1], [1,st], [st,1], [1,st], [st,1], [st,1], [st,1], [st,1], [st,1]],
        [[st,1], [st,1], [st,1], [st,1], [st,1], [1,st], [st,1], [st,1], [st,1], [st,1], [st,1]],
    ])
    samples = rand_gen(5, kernel=k, shape=k.shape[:-1])
    plt.figure()
    plt.title('Proximity Kernel')
    plt.imshow(samples[0], cmap='gray')
    # plt.show()

    # kernel from image
    img = np.array([
        [0, 0.5, 1],
        [0, 0.5, 1],
        [1, 0.5, 0]
    ])
    samples = rand_gen(5, kernel=img_to_kernel(img, 8), shape=img.shape)
    plt.figure()
    plt.title('Proximity Kernel from Image')
    plt.imshow(samples[0], cmap='gray')
    plt.show()