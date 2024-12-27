# =================================================================================================
# Refactored A3 Stereo DMap Functions + new helpers
# =================================================================================================
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

INFTY = np.inf

# =================================================================================================
# Windows Helpers: 

def SD_array(imageL, imageR, d_min, d_max):
    # Ensure inputs are NumPy arrays
    if isinstance(imageL, Image.Image):
        imageL = np.array(imageL)
    if isinstance(imageR, Image.Image):
        imageR = np.array(imageR)

    # convert images to float to prevent overflow
    imageL = imageL.astype(np.float64)
    imageR = imageR.astype(np.float64)

    # initialization of the array of "squared differences" for different shifts
    SD = np.zeros((1+d_max-d_min,np.shape(imageL)[0],np.shape(imageL)[1]))

    # compute SD for each disparity shift 
    for i, disparity in enumerate(range(d_min, d_max + 1)):
        shifted_imageR = np.roll(imageR, disparity, axis=1)            # shift the right image by the disparity

        SD[i] = np.sum((imageL - shifted_imageR) ** 2, axis=2)      # compute SD

    return SD

def integral_image(img):
    # get cumulative sum along each axis
    integral_img = img.cumsum(axis=0).cumsum(axis=1).astype(np.float64)
    return integral_img

def windSum(img, window_width):
    half_w = window_width // 2
    integral_img = integral_image(img)
    h, w = img.shape
    window_sum = np.full_like(img, INFTY, dtype=np.float64)     # pre fill with infinity
    
    # shift the integral image
    top_left = np.roll(np.roll(integral_img, -half_w, axis=0), -half_w, axis=1)
    top_right = np.roll(np.roll(integral_img, -half_w, axis=0), half_w + 1, axis=1)
    bottom_left = np.roll(np.roll(integral_img, half_w + 1, axis=0), -half_w, axis=1)
    bottom_right = np.roll(np.roll(integral_img, half_w + 1, axis=0), half_w + 1, axis=1)

    # compute the sum within each window using the integral image formula:
    #     sum = bottom-right - bottom-left - top-right + top-left
    window_sum[half_w:h - half_w, half_w:w - half_w] = (
        bottom_right[half_w:h - half_w, half_w:w - half_w] 
        - bottom_left[half_w:h - half_w, half_w:w - half_w] 
        - top_right[half_w:h - half_w, half_w:w - half_w] 
        + top_left[half_w:h - half_w, half_w:w - half_w]
    )

    return window_sum

def SSDtoDmap(SSD_array, d_min, d_max):
    # initialize map with minimum disparity
    dMap = np.full_like(SSD_array[0], d_min, dtype=np.float64)
    min_SSD = np.full_like(SSD_array[0], INFTY, dtype=np.float64)

    # for each disparity level, update the map if it has minimum SSD
    for disparity in range(d_min, d_max + 1):
        mask = SSD_array[disparity - d_min] < min_SSD
        dMap[mask] = disparity
        min_SSD[mask] = SSD_array[disparity - d_min][mask]

    # set margin disparity to zero (where INFTY is detected)
    dMap[min_SSD == INFTY] = 0
    return dMap

# =================================================================================================
# Scanlines Helpers: 

# perform the forward pass for a single scanline
def forward_pass(costs, reg_matrix):
    num_disparities, num_pixels = costs.shape
    M = np.full((num_disparities, num_pixels), np.inf) 
    M[:, 0] = costs[:, 0]
    P = np.zeros((num_disparities, num_pixels), dtype=int)  
    
    for x in range(1, num_pixels):
        total_costs = M[:, x - 1][:, np.newaxis] + reg_matrix
        M[:, x] = np.min(total_costs, axis=0) + costs[:, x]
        P[:, x] = np.argmin(total_costs, axis=0)
    
    return M, P

# backtrack to find the optimal disparities based on the matrices
def backtrack(M, P, d_minimum):
    num_pixels = M.shape[1]
    D = np.zeros(num_pixels, dtype=int)
    D[-1] = np.argmin(M[:, -1])  # start with the minimum cost at the last pixel

    # find optimal
    for x in range(num_pixels - 2, -1, -1):
        D[x] = P[D[x + 1], x + 1]
    
    # adjust to actual value
    D += d_minimum
    return D

# compute adaptive weights based on intensity differences
def compute_adaptive_weights(image, sigma):
    # convert to grayscale
    if image.ndim == 3 and image.shape[2] == 3:
        image = image.mean(axis=2)  # average RGB channels
    
    height, width = image.shape
    weights = np.zeros((height, width - 1))  # horizontal neighbors

    # iterate through each row and calculate adaptive weights for the horizontal neighbors
    for y in range(height):
        intensity_diffs = np.square(image[y, 1:] - image[y, :-1])
        weights[y, :] = np.exp(-intensity_diffs / (2 * sigma**2))

    return weights

# =================================================================================================
# MAIN FUNCTIONS: 

def Dmap_Windows(imageL, imageR, d_min, d_max, window_width):
    # Compute squared differences for all disparities
    SD = SD_array(imageL, imageR, d_min, d_max)

    # Compute SSD for each disparity with the given window size
    SSD = np.array([windSum(SD[d - d_min], window_width) for d in range(d_min, d_max + 1)])

    # Calculate disparity map from SSD values
    dMap = SSDtoDmap(SSD, d_min, d_max)
    return dMap

# computes the disparity map for a stereoscopic image pair (with the viterbi algorithm from class)
def Dmap_Viterbi(imageL, imageR, d_min, d_max, w):
    # Convert to NumPy array if input is a PIL Image
    if isinstance(imageL, Image.Image):
        imageL = np.array(imageL)
    if isinstance(imageR, Image.Image):
        imageR = np.array(imageR)

    # compute the photo-consistency term
    SSD = SD_array(imageL, imageR, d_min, d_max)
    
    # compute the regularization term
    disparities = np.arange(d_min, d_max + 1)
    reg_matrix = w * np.abs(disparities[:, None] - disparities)
    
    # initialize output
    height, width = imageL.shape[:2]
    disparity_map = np.zeros((height, width), dtype=int)
    
    # for each scan-line
    for scan_line in range(height):
        # extract photo-consistency costs for the current scanline
        costs = SSD[:, scan_line, :]
        
        # perform forward pass + backtrack
        M, P = forward_pass(costs, reg_matrix)
        disparity_map[scan_line, :] = backtrack(M, P, d_min)
    
    return disparity_map

# modified viterbi function where the photoconsistency term is smoothed using a window
def Dmap_Viterbi_Windowed(imageL, imageR, d_min, d_max, w, h):
    # Convert to NumPy array if input is a PIL Image
    if isinstance(imageL, Image.Image):
        imageL = np.array(imageL)
    if isinstance(imageR, Image.Image):
        imageR = np.array(imageR)

    # compute the photo-consistency term
    SSD = SD_array(imageL, imageR, d_min, d_max)
    num_disparities, height, width = SSD.shape

    # initialize windowed SSD array
    SSD_windowed = np.zeros_like(SSD)
    
    # apply windowed summation to each disparity level
    for d in range(num_disparities):
        smoothed_slice = windSum(SSD[d], h)
        smoothed_slice[np.isinf(smoothed_slice)] = 0    # set any INFTY values (introduced by windSum borders) to 0
        SSD_windowed[d] = smoothed_slice

    # compute the regularization term
    disparities = np.arange(d_min, d_max + 1)
    reg_matrix = w * np.abs(disparities[:, None] - disparities)

    # initialize output
    disparity_map = np.zeros((height, width), dtype=int)

    # for each scan-line
    for scan_line in range(height):
        # extract photo-consistency costs for the current scanline
        costs = SSD_windowed[:, scan_line, :]

        # perform forward pass + backtrack
        M, P = forward_pass(costs, reg_matrix)
        disparity_map[scan_line, :] = backtrack(M, P, d_min)

    return disparity_map

# modified viterbi with adaptive weights for regularization
def Dmap_Viterbi_Adaptive(imageL, imageR, d_min, d_max, w, sigma):
    # Convert to NumPy array if input is a PIL Image
    if isinstance(imageL, Image.Image):
        imageL = np.array(imageL)
    if isinstance(imageR, Image.Image):
        imageR = np.array(imageR)
        
    # compute the photo-consistency term
    SSD = SD_array(imageL, imageR, d_min, d_max)
    num_disparities, height, width = SSD.shape

    # compute the regularization term
    disparities = np.arange(d_min, d_max + 1)
    disparity_map = np.zeros((height, width), dtype=int)

    # compute adaptive weights for intensity differences in the left image
    adaptive_weights = compute_adaptive_weights(imageL, sigma)

    # for each scan-line
    for scan_line in range(height):
        # extract photo-consistency costs for the current scanline
        costs = SSD[:, scan_line, :]

        # perform forward pass applying the adaptive weights:

        # create regularization matrix for this scanline with adaptive weights
        M = np.full((num_disparities, width), np.inf) 
        M[:, 0] = costs[:, 0]
        P = np.zeros((num_disparities, width), dtype=int)
        
        for x in range(1, width):
            # apply adaptive weight for this position
            weight = adaptive_weights[scan_line, x - 1] * w
            
            # regularization costs for current column x with adjusted weights
            reg_matrix = weight * np.abs(disparities[:, None] - disparities)
            
            # calculate total costs with adaptive regularization
            total_costs = M[:, x - 1][:, np.newaxis] + reg_matrix
            M[:, x] = np.min(total_costs, axis=0) + costs[:, x]
            P[:, x] = np.argmin(total_costs, axis=0)

        # perform normal backtrack
        disparity_map[scan_line, :] = backtrack(M, P, d_min)

    return disparity_map

# =================================================================================================
# Visualization: 

def compare_disparity_maps(dmap1, dmap2, dmap3, dmap4):
    titles=["Window-based", "Viterbi", "Viterbi + Windowed", "Viterbi + Adaptive"]

    fig, axs = plt.subplots(2, 2, figsize=(12, 5.5))

    # Compute global min and max for the color bar
    vmin = min(map(np.min, [dmap1, dmap2, dmap3, dmap4]))
    vmax = max(map(np.max, [dmap1, dmap2, dmap3, dmap4]))

    # Display each disparity map with its title
    disparity_maps = [dmap1, dmap2, dmap3, dmap4]
    for i, ax in enumerate(axs.flat):
        im = ax.imshow(disparity_maps[i], cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(titles[i], fontsize=12)
        ax.axis('off')

    # Add a universal color bar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Disparity', fontsize=12)

    # Add the aggregate title
    fig.suptitle("Comparison of Left to Right Disparity Maps", fontsize=16, weight='bold')

    plt.tight_layout(rect=[0, 0.03, 0.9, 1.0])
    plt.show()

# =================================================================================================
# convert output to torch tensor: 

def make_dmap_tensor(disparity_map):
    """
    Prepares the disparity map for compatibility with dmap_wind_loss.
    Args:
        disparity_map (numpy.ndarray or torch.Tensor): Input disparity map.
    Returns:
        torch.Tensor: Processed disparity map.
    """
    disparity_map = torch.tensor(disparity_map, dtype=torch.float32)
    
    # Ensure it has a batch and channel dimension
    disparity_map = disparity_map.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # Move to the correct device
    return disparity_map.to(torch.device("cpu"))


