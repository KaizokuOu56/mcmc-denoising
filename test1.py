import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
import time

class MCMCDDenoiser:
    """
    Markov-chain Monte Carlo Denoising (MCMCD) implementation
    Based on Wong et al. (2011) paper
    """
    
    def __init__(self, sigma_s=21, region_radius=3, num_iterations=200):
        """
        Initialize MCMCD denoiser
        
        Parameters:
        -----------
        sigma_s : float
            Spatial variance for instrumental distribution (default: 21)
        region_radius : int
            Radius of local neighborhood (default: 3)
        num_iterations : int
            Number of MCMC iterations (default: 200)
        """
        self.sigma_s = sigma_s
        self.region_radius = region_radius
        self.num_iterations = num_iterations
        
    def _get_local_region(self, image, s, radius):
        """Extract local circular region around site s"""
        h, w = image.shape
        y, x = s
        
        # Create circular mask
        y_coords, x_coords = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = x_coords**2 + y_coords**2 <= radius**2
        
        # Extract region with boundary handling
        y_min = max(0, y - radius)
        y_max = min(h, y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(w, x + radius + 1)
        
        region = image[y_min:y_max, x_min:x_max]
        
        # Adjust mask for boundary cases
        mask_y_min = radius - (y - y_min)
        mask_y_max = mask_y_min + (y_max - y_min)
        mask_x_min = radius - (x - x_min)
        mask_x_max = mask_x_min + (x_max - x_min)
        
        adjusted_mask = mask[mask_y_min:mask_y_max, mask_x_min:mask_x_max]
        
        return region[adjusted_mask]
    
    def _compute_objective(self, F, s0, sk, sigma_n, sigma_l):
        """
        Compute Geman-McClure objective function (Eq. 9)
        
        Parameters:
        -----------
        F : ndarray
            Noisy image
        s0 : tuple
            Reference site (y, x)
        sk : tuple
            Candidate site (y, x)
        sigma_n : float
            Noise variance
        sigma_l : float
            Local variance
        """
        # Get local regions
        region_s0 = self._get_local_region(F, s0, self.region_radius)
        region_sk = self._get_local_region(F, sk, self.region_radius)
        
        # Handle different region sizes due to boundary effects
        min_len = min(len(region_s0), len(region_sk))
        region_s0 = region_s0[:min_len]
        region_sk = region_sk[:min_len]
        
        # Compute Geman-McClure error statistics
        diff = region_sk - region_s0
        diff_sq = diff ** 2
        
        # Avoid division by zero
        denominator = sigma_n**4 * sigma_l**2 + diff_sq
        denominator = np.maximum(denominator, 1e-10)
        
        exponent = -np.sum(diff_sq / denominator)
        
        return np.exp(exponent)
    
    def _estimate_noise_variance(self, image):
        """Estimate noise variance using median absolute deviation"""
        # Use high-frequency component estimation
        diff_h = np.diff(image, axis=0)
        diff_v = np.diff(image, axis=1)
        sigma = np.median(np.abs(np.concatenate([diff_h.flatten(), diff_v.flatten()]))) / 0.6745
        return sigma ** 2
    
    def _estimate_local_variance(self, image, s):
        """Estimate local variance around site s"""
        region = self._get_local_region(image, s, self.region_radius)
        return np.var(region) if len(region) > 0 else 1.0
    
    def denoise(self, noisy_image, sigma_n=None, verbose=False):
        """
        Denoise image using MCMCD
        
        Parameters:
        -----------
        noisy_image : ndarray
            Noisy input image
        sigma_n : float, optional
            Known noise standard deviation (if None, will be estimated)
        verbose : bool
            Print progress information
        
        Returns:
        --------
        denoised_image : ndarray
            Denoised image
        """
        F = noisy_image.copy()
        h, w = F.shape
        
        # Estimate noise variance if not provided
        if sigma_n is None:
            sigma_n_var = self._estimate_noise_variance(F)
            sigma_n = np.sqrt(sigma_n_var)
        else:
            sigma_n_var = sigma_n ** 2
        
        if verbose:
            print(f"Image size: {h}x{w}")
            print(f"Noise std: {sigma_n:.2f}")
            print(f"MCMC iterations per pixel: {self.num_iterations}")
            print(f"Processing pixels...")
        
        denoised = np.zeros_like(F)
        
        # Process each pixel with progress bar
        total_pixels = h * w
        pixel_iterator = tqdm(np.ndindex(h, w), 
                             total=total_pixels,
                             desc="Denoising",
                             unit="pixels",
                             disable=not verbose,
                             ncols=80)
        
        for (i, j) in pixel_iterator:
            s0 = (i, j)
            sigma_l = np.sqrt(self._estimate_local_variance(F, s0))
            
            # MCMC sampling
            samples = []
            weights = []
            
            s_prev = s0
            
            k = 0
            while(k < self.num_iterations):
                # Draw candidate sample from instrumental distribution (Gaussian)
                dy = np.random.normal(0, self.sigma_s)
                dx = np.random.normal(0, self.sigma_s)
                
                sk_y = int(np.clip(s_prev[0] + dy, 0, h - 1))
                sk_x = int(np.clip(s_prev[1] + dx, 0, w - 1))
                sk = (sk_y, sk_x)
                
                # Compute acceptance probability
                f_sk = self._compute_objective(F, s0, sk, sigma_n, sigma_l)
                f_prev = self._compute_objective(F, s0, s_prev, sigma_n, sigma_l)
                
                # Avoid division by zero
                if f_prev > 0:
                    acceptance_prob = min(1.0, f_sk / f_prev)
                else:
                    acceptance_prob = 1.0
                
                # Accept or reject
                u = np.random.uniform(0, 1)
                if u <= acceptance_prob:
                    samples.append(sk)
                    weights.append(f_sk)
                    s_prev = sk
                    k += 1
            
            # Estimate posterior and compute conditional mean
            if len(samples) > 0 and np.sum(weights) > 0:
                # Importance-weighted posterior estimation
                intensities = [F[s] for s in samples]
                weights = np.array(weights)
                weights = weights / np.sum(weights)  # Normalize
                
                # Conditional mean
                denoised[i, j] = np.sum(intensities * weights)
            else:
                # Fallback to original value if no samples
                denoised[i, j] = F[i, j]
        
        if verbose:
            print("\nDenoising complete!")
        
        return denoised
    
def generate_test_image(size=128, pattern='camera'):
    """
    Generate test images from scipy.datasets and other sources
    
    Parameters:
    -----------
    size : int
        Image size (will be resized to size x size)
    pattern : str
        Image type: 'camera', 'astronaut', 'coins', 'moon', 'text', 'checkerboard'
    """
    from scipy import datasets, ndimage
    from skimage import transform, color
    
    if pattern == 'camera':
        # Camera image (512x512 grayscale)
        image = datasets.face(gray=True)
        
    elif pattern == 'astronaut':
        # Astronaut image (needs conversion to grayscale)
        image = datasets.ascent()
        
    elif pattern == 'coins':
        # Generate coins-like image using scipy
        from scipy import ndimage
        # Create a synthetic coins pattern
        image = np.zeros((size, size))
        y, x = np.ogrid[:size, :size]
        num_coins = 8
        for _ in range(num_coins):
            cy = np.random.randint(size//8, 7*size//8)
            cx = np.random.randint(size//8, 7*size//8)
            radius = np.random.randint(size//16, size//10)
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            image[mask] = np.random.uniform(150, 220)
        # Add some texture
        image = ndimage.gaussian_filter(image, sigma=1.5)
        image = np.clip(image * 255 / image.max(), 0, 255)
        return image
        
    elif pattern == 'moon':
        # Create moon-like surface
        from scipy import ndimage
        image = np.random.rand(size, size) * 255
        image = ndimage.gaussian_filter(image, sigma=3)
        # Add craters
        y, x = np.ogrid[:size, :size]
        for _ in range(10):
            cy = np.random.randint(0, size)
            cx = np.random.randint(0, size)
            radius = np.random.randint(size//20, size//8)
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            image[mask] *= 0.6
        image = np.clip(image, 0, 255)
        return image
        
    elif pattern == 'text':
        # Text-like pattern
        image = np.ones((size, size)) * 255
        # Create horizontal lines (text-like)
        line_height = size // 20
        for i in range(size//10, size, size//8):
            thickness = np.random.randint(2, 5)
            image[i:i+thickness, size//10:9*size//10] = np.random.uniform(0, 50)
            # Add some "words" as blocks
            for j in range(size//10, 9*size//10, size//6):
                word_len = np.random.randint(size//15, size//8)
                image[i:i+thickness, j:j+word_len] = np.random.uniform(0, 50)
        return image
        
    elif pattern == 'checkerboard':
        # Checkerboard pattern
        from scipy import datasets
        image = np.zeros((size, size))
        square_size = size // 8
        for i in range(0, size, square_size):
            for j in range(0, size, square_size):
                if ((i // square_size) + (j // square_size)) % 2 == 0:
                    image[i:i+square_size, j:j+square_size] = 200
        return image
    
    else:
        # Default to ascent image
        image = datasets.ascent()
    
    # Resize to desired size
    if image.shape != (size, size):
        from skimage import transform
        image = transform.resize(image, (size, size), anti_aliasing=True, preserve_range=True)
    
    return np.clip(image, 0, 255)


def add_gaussian_noise(image, sigma):
    """Add Gaussian noise to image"""
    noise = np.random.normal(0, sigma, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255)


def test_denoising(image_size=128, noise_sigma=20, pattern='camera'):
    """
    Test MCMCD denoising on test image
    
    Parameters:
    -----------
    image_size : int
        Size of test image
    noise_sigma : float
        Standard deviation of Gaussian noise
    pattern : str
        Image type: 'camera', 'astronaut', 'coins', 'moon', 'text', 'checkerboard'
    """
    print(f"\n{'='*60}")
    print(f"MCMCD Denoising Test")
    print(f"{'='*60}")
    
    # Generate original image
    print(f"\n1. Generating {pattern} test image ({image_size}x{image_size})...")
    original = generate_test_image(image_size, pattern)
    
    # Add noise
    print(f"2. Adding Gaussian noise (σ = {noise_sigma})...")
    noisy = add_gaussian_noise(original, noise_sigma)
    
    # Denoise
    print(f"3. Denoising with MCMCD...")
    denoiser = MCMCDDenoiser(sigma_s=21, region_radius=3, num_iterations=200)
    
    start_time = time.time()
    denoised = denoiser.denoise(noisy, sigma_n=noise_sigma, verbose=True)
    elapsed_time = time.time() - start_time
    
    # Calculate metrics
    psnr_noisy = peak_signal_noise_ratio(original, noisy, data_range=255)
    psnr_denoised = peak_signal_noise_ratio(original, denoised, data_range=255)
    
    ssim_noisy = structural_similarity(original, noisy, data_range=255)
    ssim_denoised = structural_similarity(original, denoised, data_range=255)
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"{'='*60}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"\nNoisy Image:")
    print(f"  PSNR: {psnr_noisy:.2f} dB")
    print(f"  SSIM: {ssim_noisy:.4f}")
    print(f"\nDenoised Image:")
    print(f"  PSNR: {psnr_denoised:.2f} dB")
    print(f"  SSIM: {ssim_denoised:.4f}")
    print(f"\nImprovement:")
    print(f"  ΔPSNR: +{psnr_denoised - psnr_noisy:.2f} dB")
    print(f"  ΔSSIM: +{ssim_denoised - ssim_noisy:.4f}")
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(noisy, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(f'Noisy Image\nPSNR: {psnr_noisy:.2f} dB, SSIM: {ssim_noisy:.4f}', 
                      fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(denoised, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title(f'Denoised Image (MCMCD)\nPSNR: {psnr_denoised:.2f} dB, SSIM: {ssim_denoised:.4f}', 
                      fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return original, noisy, denoised


# Example usage
if __name__ == "__main__":
    # Test with different images and noise levels
    print("Running MCMCD Denoising Tests\n")
    
    # Test 1: Camera/Face image with moderate noise
    test_denoising(image_size=128, noise_sigma=20, pattern='camera')
    
    # Uncomment to test other images:
    # test_denoising(image_size=128, noise_sigma=25, pattern='astronaut')
    # test_denoising(image_size=128, noise_sigma=20, pattern='coins')
    # test_denoising(image_size=128, noise_sigma=30, pattern='moon')
    # test_denoising(image_size=128, noise_sigma=20, pattern='text')
    # test_denoising(image_size=128, noise_sigma=25, pattern='checkerboard')