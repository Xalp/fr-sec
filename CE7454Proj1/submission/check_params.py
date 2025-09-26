"""
Calculate model parameters without requiring PyTorch installation
Based on the architecture in model.py
"""

def calculate_conv_params(in_channels, out_channels, kernel_size, bias=True):
    """Calculate parameters for a Conv2d layer"""
    params = in_channels * out_channels * kernel_size * kernel_size
    if bias:
        params += out_channels
    return params

def calculate_bn_params(channels):
    """Calculate parameters for BatchNorm2d (2 * channels for weight and bias)"""
    return 2 * channels

def calculate_unet_conv2_params(in_size, out_size, is_batchnorm=True):
    """Calculate parameters for unetConv2 block"""
    params = 0
    # First conv
    params += calculate_conv_params(in_size, out_size, 3)
    if is_batchnorm:
        params += calculate_bn_params(out_size)
    # Second conv
    params += calculate_conv_params(out_size, out_size, 3)
    if is_batchnorm:
        params += calculate_bn_params(out_size)
    return params

def calculate_attention_gate_params(F_g, F_l, F_int):
    """Calculate parameters for AttentionGate"""
    params = 0
    # W_g
    params += calculate_conv_params(F_g, F_int, 1)
    params += calculate_bn_params(F_int)
    # W_x
    params += calculate_conv_params(F_l, F_int, 1)
    params += calculate_bn_params(F_int)
    # psi
    params += calculate_conv_params(F_int, 1, 1)
    params += calculate_bn_params(1)
    return params

def calculate_unet_up_params(in_size, out_size, is_deconv=True, is_batchnorm=True):
    """Calculate parameters for unetUp block"""
    params = 0
    # Conv block
    params += calculate_unet_conv2_params(in_size, out_size, is_batchnorm)
    # Attention gate
    params += calculate_attention_gate_params(in_size//2, in_size//2, in_size//4)
    # Upsampling
    if is_deconv:
        params += calculate_conv_params(in_size, out_size, 2, bias=True)
    # Note: Bilinear upsampling has no parameters
    return params

def calculate_attention_unet_params(feature_scale=2, n_classes=19, in_channels=3, is_batchnorm=True, is_deconv=True):
    """Calculate total parameters for AttentionUNet"""
    filters = [64, 128, 256, 512, 1024]
    filters = [int(x / feature_scale) for x in filters]
    
    total_params = 0
    
    # Encoder
    total_params += calculate_unet_conv2_params(in_channels, filters[0], is_batchnorm)
    total_params += calculate_unet_conv2_params(filters[0], filters[1], is_batchnorm)
    total_params += calculate_unet_conv2_params(filters[1], filters[2], is_batchnorm)
    total_params += calculate_unet_conv2_params(filters[2], filters[3], is_batchnorm)
    
    # Center
    total_params += calculate_unet_conv2_params(filters[3], filters[4], is_batchnorm)
    
    # Decoder with attention
    total_params += calculate_unet_up_params(filters[4], filters[3], is_deconv, is_batchnorm)
    total_params += calculate_unet_up_params(filters[3], filters[2], is_deconv, is_batchnorm)
    total_params += calculate_unet_up_params(filters[2], filters[1], is_deconv, is_batchnorm)
    total_params += calculate_unet_up_params(filters[1], filters[0], is_deconv, is_batchnorm)
    
    # Final conv
    total_params += calculate_conv_params(filters[0], n_classes, 1)
    
    return total_params

# Test with different feature scales
for scale in [2, 4, 6, 8, 10]:
    params = calculate_attention_unet_params(feature_scale=scale)
    filters = [64, 128, 256, 512, 1024]
    filters = [int(x / scale) for x in filters]
    print(f"\nfeature_scale={scale}: filters={filters}")
    print(f"Parameters: {params:,}")
    print(f"Within limit: {params <= 1821085}")