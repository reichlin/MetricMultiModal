import numpy as np
# import cv2


def gaussian_noise(x, mode, _, __, configs_noises):
    x = x[mode]

    mean = configs_noises['gaussian_noise']['mean'][mode] #0.0
    std = configs_noises['gaussian_noise']['std'][mode] #0.1

    noise = np.random.normal(mean, std, x.shape) if mode == 'sound' else np.random.normal(mean, std, x.shape).astype(x.dtype)
    return x + noise


def salt_and_pepper_noise(x, mode, _, __, configs_noises):
    x = x[mode]

    ratio = configs_noises['salt_and_pepper_noise']['ratio']
    val_max = np.array(configs_noises['salt_and_pepper_noise']['max'][mode])
    val_min = -val_max if (mode == 'state' or mode == 'sound') else 0

    salt = (np.random.rand(*x.shape) < (ratio / 2)) * val_max
    pepper = (np.random.rand(*x.shape) > (ratio / 2)) * 1.0

    return np.clip((x + salt) * pepper, a_min=val_min, a_max=val_max).astype(x.dtype)


def patches_noise(x, mode, _, __, configs_noises):
    x = x[mode]

    patch_ratio = configs_noises['patches_noise']['patch_ratio'] #0.3

    H, W = x.shape[-2], x.shape[-1]

    patch_size = int(H * patch_ratio)

    top_corner_x = np.random.randint(0, H-patch_size)
    top_corner_y = np.random.randint(0, W-patch_size)

    mask = np.ones((H, W), dtype=np.uint8)
    mask[top_corner_x:top_corner_x+patch_size, top_corner_y:top_corner_y+patch_size] = 0

    return x * mask


def puzzle_noise(x, mode, _, __, configs_noises):
    x = x[mode]

    n_patches = configs_noises['puzzle_noise']['n_patches'] #3

    H, W = x.shape[-2], x.shape[-1]

    patch_h, patch_w = H // n_patches, W // n_patches
    patches = [
        x[:, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w]
        for i in range(n_patches) for j in range(n_patches)
    ]

    np.random.shuffle(patches)

    shuffled_x = np.zeros_like(x)
    for idx, patch in enumerate(patches):
        i, j = divmod(idx, n_patches)
        shuffled_x[:, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w] = patch

    return shuffled_x


def sensor_failure(x, mode, _, __, configs_noises):
    x = x[mode]
    return x * 0
    # if type(x) == np.ndarray:
    #     return x * 0
    # else:
    #     return [xi * 0 for xi in x]


def texture_noise(x, mode, all_bg_imgs, _, configs_noises):
    x = x[mode]

    min_depth = configs_noises['texture_noise']['min_depth']

    background = all_bg_imgs['bg_imgs'][np.random.randint(0, all_bg_imgs['bg_imgs'].shape[0])]

    if 'depth' in all_bg_imgs.keys():
        depth = all_bg_imgs['depth']
        mask = np.expand_dims(depth < depth[0, 0] * min_depth, 0)
    else:
        mask = (x.mean(0) < 254)

    x_foreground = x * mask
    x_background = background * (1 - mask)
    rescale_term = 1.0 if x_background.max() > 1.0 else 255
    x_background = (np.concatenate([x_background] * (x_foreground.shape[0] // x_background.shape[0]), 0) * rescale_term).astype(np.uint8)

    return x_foreground + x_background


def hallucination_noise(x, mode, _, init_obs, configs_noises):
    return init_obs[mode]


