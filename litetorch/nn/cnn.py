"""
Convolutional Neural Network (CNN) layers and architectures.

This module contains templates for implementing CNN components including
convolutional layers, pooling layers, batch normalization, and common
CNN architectures like ResNet, VGG, etc.
"""
import numpy as np


class Conv2d:
    """
    2D Convolutional layer.
    
    Applies a 2D convolution over an input signal composed of several input planes.
    This is the fundamental building block of convolutional neural networks (CNNs).
    
    The layer creates a convolution kernel that is convolved with the input to
    produce a tensor of outputs. If bias is enabled, a bias vector is created
    and added to the outputs.
    
    Mathematical operation:
        output[b, c_out, h_out, w_out] = bias[c_out] + 
            sum_{c_in, kh, kw} weight[c_out, c_in, kh, kw] * 
            input[b, c_in, h_in + kh, w_in + kw]
    
    References:
        - LeCun et al. (1989): "Backpropagation Applied to Handwritten Zip Code Recognition"
        - Krizhevsky et al. (2012): "ImageNet Classification with Deep CNNs" (AlexNet)
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True):
        """
        Initialize Conv2d layer.
        
        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels produced by the convolution
            kernel_size: Size of the convolving kernel (int or tuple)
            stride: Stride of the convolution (default: 1)
            padding: Zero-padding added to both sides of the input (default: 0)
            dilation: Spacing between kernel elements (default: 1)
            groups: Number of blocked connections from input to output (default: 1)
            bias: If True, adds a learnable bias to the output (default: True)
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        
        # Initialize weights and biases
        self.weight = self._init_weights()
        self.bias = self._init_bias() if bias else None
    
    def _init_weights(self):
        """
        Initialize convolutional weights using He/Kaiming initialization.
        
        He initialization is appropriate for ReLU activations, scaling weights
        by sqrt(2 / fan_in) where fan_in is the number of input units.
        
        Returns:
            Weight tensor of shape (out_channels, in_channels, kernel_h, kernel_w)
        """
        kh, kw = self.kernel_size
        fan_in = self.in_channels * kh * kw
        scale = np.sqrt(2.0 / fan_in)
        shape = (self.out_channels, self.in_channels // self.groups, kh, kw)
        return np.random.randn(*shape) * scale
    
    def _init_bias(self):
        """
        Initialize bias to zeros.
        
        Returns:
            Bias vector of shape (out_channels,)
        """
        return np.zeros(self.out_channels)
    
    def forward(self, x):
        """
        Forward pass of the convolutional layer.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        
        Returns:
            Output tensor of shape (batch_size, out_channels, out_height, out_width)
            where out_height and out_width depend on padding, stride, and kernel size:
            out_height = (height + 2*padding - dilation*(kernel_h-1) - 1) / stride + 1
            out_width = (width + 2*padding - dilation*(kernel_w-1) - 1) / stride + 1
        """
        # TODO: Implement forward pass
        # 1. Apply padding to input if needed
        # 2. Extract image patches (im2col operation)
        # 3. Perform convolution via matrix multiplication
        # 4. Reshape output to proper dimensions
        # 5. Add bias if enabled
        pass
    
    def _im2col(self, x):
        """
        Transform input for efficient convolution using im2col algorithm.
        
        The im2col (image to column) operation rearranges image patches into
        columns, allowing convolution to be computed as a matrix multiplication.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        
        Returns:
            Column matrix suitable for matrix multiplication
        """
        # TODO: Implement im2col transformation
        pass


class MaxPool2d:
    """
    2D Max Pooling layer.
    
    Applies a 2D max pooling over an input signal. Max pooling reduces the
    spatial dimensions by taking the maximum value in each local region,
    providing translation invariance and reducing computation.
    
    References:
        - LeCun et al. (1998): "Gradient-Based Learning Applied to Document Recognition"
    """
    
    def __init__(self, kernel_size, stride=None, padding=0):
        """
        Initialize MaxPool2d layer.
        
        Args:
            kernel_size: Size of the pooling window (int or tuple)
            stride: Stride of the pooling operation (default: kernel_size)
            padding: Zero-padding added to both sides (default: 0)
        """
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride if stride is not None else self.kernel_size
        self.padding = padding
    
    def forward(self, x):
        """
        Forward pass of the max pooling layer.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Output tensor of shape (batch_size, channels, out_height, out_width)
            where out_height and out_width are reduced by the pooling operation
        """
        # TODO: Implement forward pass
        # 1. Apply padding if needed
        # 2. Slide pooling window over input
        # 3. Take maximum value in each window
        # 4. Store indices for backward pass (needed for unpooling)
        pass
    
    def backward(self, grad_output):
        """
        Backward pass for max pooling.
        
        Gradients only flow through the maximum values selected during forward pass.
        
        Args:
            grad_output: Gradient from the next layer
        
        Returns:
            Gradient with respect to input
        """
        # TODO: Implement backward pass
        pass


class AvgPool2d:
    """
    2D Average Pooling layer.
    
    Applies a 2D average pooling over an input signal. Average pooling reduces
    spatial dimensions by averaging values in each local region.
    """
    
    def __init__(self, kernel_size, stride=None, padding=0):
        """
        Initialize AvgPool2d layer.
        
        Args:
            kernel_size: Size of the pooling window (int or tuple)
            stride: Stride of the pooling operation (default: kernel_size)
            padding: Zero-padding added to both sides (default: 0)
        """
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride if stride is not None else self.kernel_size
        self.padding = padding
    
    def forward(self, x):
        """
        Forward pass of the average pooling layer.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Output tensor with reduced spatial dimensions
        """
        # TODO: Implement forward pass
        # Similar to MaxPool2d but compute average instead of max
        pass


class BatchNorm2d:
    """
    2D Batch Normalization layer.
    
    Normalizes the input by subtracting the batch mean and dividing by the
    batch standard deviation. This helps with training stability and allows
    higher learning rates.
    
    During training: normalizes using batch statistics
    During inference: uses running statistics computed during training
    
    BN(x) = gamma * (x - mean) / sqrt(var + eps) + beta
    
    References:
        - Ioffe & Szegedy (2015): "Batch Normalization: Accelerating Deep Network 
          Training by Reducing Internal Covariate Shift"
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        """
        Initialize BatchNorm2d layer.
        
        Args:
            num_features: Number of channels (C) from input of size (N, C, H, W)
            eps: Small constant for numerical stability (default: 1e-5)
            momentum: Momentum for running mean/var update (default: 0.1)
            affine: If True, learn scale (gamma) and shift (beta) parameters
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.training = True
        
        # Learnable parameters
        if affine:
            self.gamma = np.ones(num_features)  # Scale parameter
            self.beta = np.zeros(num_features)   # Shift parameter
        else:
            self.gamma = None
            self.beta = None
        
        # Running statistics (for inference)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def forward(self, x):
        """
        Forward pass of batch normalization.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Normalized tensor of same shape as input
        """
        # TODO: Implement forward pass
        # Training mode:
        # 1. Compute batch mean and variance across (N, H, W) dimensions
        # 2. Normalize: x_norm = (x - mean) / sqrt(var + eps)
        # 3. Scale and shift: y = gamma * x_norm + beta
        # 4. Update running statistics
        #
        # Inference mode:
        # 1. Use running_mean and running_var for normalization
        # 2. Apply scale and shift
        pass
    
    def train(self):
        """Set the module in training mode."""
        self.training = True
    
    def eval(self):
        """Set the module in evaluation mode."""
        self.training = False


class Dropout2d:
    """
    2D Dropout layer for CNNs.
    
    Randomly zeros out entire channels of the input tensor with probability p.
    This is more appropriate for convolutional layers than regular dropout,
    as it drops entire feature maps rather than individual elements.
    
    References:
        - Tompson et al. (2015): "Efficient Object Localization Using CNNs"
    """
    
    def __init__(self, p=0.5):
        """
        Initialize Dropout2d layer.
        
        Args:
            p: Probability of dropping a channel (default: 0.5)
        """
        self.p = p
        self.training = True
    
    def forward(self, x):
        """
        Forward pass of dropout.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Tensor with randomly dropped channels during training,
            unchanged tensor during evaluation
        """
        if not self.training or self.p == 0:
            return x
        
        # TODO: Implement channel-wise dropout
        # 1. Create mask of shape (batch_size, channels, 1, 1)
        # 2. Sample from Bernoulli distribution
        # 3. Apply mask and scale by 1/(1-p)
        pass
    
    def train(self):
        """Set the module in training mode."""
        self.training = True
    
    def eval(self):
        """Set the module in evaluation mode."""
        self.training = False


# Common CNN Architectures

class VGGBlock:
    """
    VGG-style convolutional block.
    
    Consists of multiple Conv2d layers with ReLU activations,
    followed by max pooling.
    
    References:
        - Simonyan & Zisserman (2014): "Very Deep Convolutional Networks 
          for Large-Scale Image Recognition"
    """
    
    def __init__(self, in_channels, out_channels, num_convs=2):
        """
        Initialize VGG block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_convs: Number of convolutional layers (default: 2)
        """
        self.convs = []
        for i in range(num_convs):
            self.convs.append(Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ))
        self.pool = MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        """
        Forward pass through VGG block.
        
        Args:
            x: Input tensor
        
        Returns:
            Output after convolutions, ReLU, and pooling
        """
        # TODO: Implement forward pass
        # For each conv layer:
        #   x = conv(x)
        #   x = relu(x)
        # x = pool(x)
        pass


class ResidualBlock:
    """
    Residual block for ResNet architectures.
    
    Implements a residual connection (skip connection) that allows
    gradients to flow through the network more easily, enabling
    very deep networks.
    
    output = F(x) + x
    
    where F(x) is the residual function (conv-bn-relu-conv-bn).
    
    References:
        - He et al. (2016): "Deep Residual Learning for Image Recognition"
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Initialize Residual block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for the first convolution (default: 1)
        """
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, 
                           stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(out_channels)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3,
                           stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1,
                                  stride=stride, bias=False)
            self.shortcut_bn = BatchNorm2d(out_channels)
    
    def forward(self, x):
        """
        Forward pass through residual block.
        
        Args:
            x: Input tensor
        
        Returns:
            Output with residual connection: F(x) + x
        """
        # TODO: Implement forward pass
        # identity = x
        # out = relu(bn1(conv1(x)))
        # out = bn2(conv2(out))
        # if shortcut exists:
        #     identity = shortcut_bn(shortcut(x))
        # out = out + identity
        # out = relu(out)
        pass


class BottleneckBlock:
    """
    Bottleneck block for deeper ResNet architectures (ResNet-50, 101, 152).
    
    Uses 1x1 convolutions to reduce dimensions, followed by 3x3 convolution,
    and 1x1 convolution to restore dimensions. This is more efficient for
    very deep networks.
    
    Architecture: 1x1 conv -> 3x3 conv -> 1x1 conv with skip connection
    
    References:
        - He et al. (2016): "Deep Residual Learning for Image Recognition"
    """
    
    def __init__(self, in_channels, bottleneck_channels, out_channels, stride=1):
        """
        Initialize Bottleneck block.
        
        Args:
            in_channels: Number of input channels
            bottleneck_channels: Number of channels in the bottleneck (typically out_channels/4)
            out_channels: Number of output channels
            stride: Stride for the 3x3 convolution (default: 1)
        """
        # 1x1 conv to reduce dimensions
        self.conv1 = Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(bottleneck_channels)
        
        # 3x3 conv
        self.conv2 = Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3,
                           stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(bottleneck_channels)
        
        # 1x1 conv to restore dimensions
        self.conv3 = Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1,
                                  stride=stride, bias=False)
            self.shortcut_bn = BatchNorm2d(out_channels)
    
    def forward(self, x):
        """
        Forward pass through bottleneck block.
        
        Args:
            x: Input tensor
        
        Returns:
            Output with residual connection
        """
        # TODO: Implement forward pass similar to ResidualBlock
        pass


class InceptionModule:
    """
    Inception module (Naive version) from GoogLeNet.
    
    Applies multiple convolutional filters with different kernel sizes
    in parallel and concatenates the results. This allows the network
    to learn features at multiple scales.
    
    Parallel branches:
    - 1x1 convolution
    - 3x3 convolution (with 1x1 reduction)
    - 5x5 convolution (with 1x1 reduction)
    - 3x3 max pooling (with 1x1 projection)
    
    References:
        - Szegedy et al. (2015): "Going Deeper with Convolutions"
    """
    
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        """
        Initialize Inception module.
        
        Args:
            in_channels: Number of input channels
            ch1x1: Output channels for 1x1 conv branch
            ch3x3red: Reduction channels for 3x3 conv branch
            ch3x3: Output channels for 3x3 conv branch
            ch5x5red: Reduction channels for 5x5 conv branch
            ch5x5: Output channels for 5x5 conv branch
            pool_proj: Output channels for pooling branch projection
        """
        # 1x1 conv branch
        self.branch1 = Conv2d(in_channels, ch1x1, kernel_size=1)
        
        # 3x3 conv branch
        self.branch2_1 = Conv2d(in_channels, ch3x3red, kernel_size=1)
        self.branch2_2 = Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        
        # 5x5 conv branch
        self.branch3_1 = Conv2d(in_channels, ch5x5red, kernel_size=1)
        self.branch3_2 = Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        
        # Pooling branch
        self.branch4_1 = MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_2 = Conv2d(in_channels, pool_proj, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass through inception module.
        
        Args:
            x: Input tensor
        
        Returns:
            Concatenated output from all branches
        """
        # TODO: Implement forward pass
        # branch1 = relu(conv1x1(x))
        # branch2 = relu(conv3x3(relu(conv1x1_reduce(x))))
        # branch3 = relu(conv5x5(relu(conv1x1_reduce(x))))
        # branch4 = relu(conv1x1_proj(pool(x)))
        # output = concatenate([branch1, branch2, branch3, branch4], axis=channels)
        pass


class DepthwiseSeparableConv2d:
    """
    Depthwise Separable Convolution.
    
    Factorizes a standard convolution into a depthwise convolution and a
    pointwise (1x1) convolution, significantly reducing parameters and
    computation while maintaining similar performance.
    
    Used in efficient architectures like MobileNet and EfficientNet.
    
    References:
        - Howard et al. (2017): "MobileNets: Efficient CNNs for Mobile Vision Applications"
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        Initialize Depthwise Separable Convolution.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolving kernel
            stride: Stride of the convolution (default: 1)
            padding: Zero-padding added to both sides (default: 0)
        """
        # Depthwise convolution (one filter per input channel)
        self.depthwise = Conv2d(in_channels, in_channels, kernel_size,
                               stride=stride, padding=padding, groups=in_channels)
        self.bn1 = BatchNorm2d(in_channels)
        
        # Pointwise convolution (1x1 conv to combine channels)
        self.pointwise = Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn2 = BatchNorm2d(out_channels)
    
    def forward(self, x):
        """
        Forward pass through depthwise separable convolution.
        
        Args:
            x: Input tensor
        
        Returns:
            Output after depthwise and pointwise convolutions
        """
        # TODO: Implement forward pass
        # x = relu(bn1(depthwise(x)))
        # x = relu(bn2(pointwise(x)))
        pass


# Complete CNN Architectures

class LeNet5:
    """
    LeNet-5 architecture for digit recognition.
    
    One of the pioneering CNN architectures, designed for handwritten
    digit recognition on the MNIST dataset.
    
    Architecture:
        Input (32x32) -> Conv(6) -> AvgPool -> Conv(16) -> AvgPool -> 
        FC(120) -> FC(84) -> FC(10)
    
    References:
        - LeCun et al. (1998): "Gradient-Based Learning Applied to Document Recognition"
    """
    
    def __init__(self, num_classes=10):
        """
        Initialize LeNet-5.
        
        Args:
            num_classes: Number of output classes (default: 10 for MNIST)
        """
        self.conv1 = Conv2d(1, 6, kernel_size=5, stride=1)
        self.pool1 = AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = Conv2d(6, 16, kernel_size=5, stride=1)
        self.pool2 = AvgPool2d(kernel_size=2, stride=2)
        # TODO: Add fully connected layers
        self.fc1 = None  # 16*5*5 -> 120
        self.fc2 = None  # 120 -> 84
        self.fc3 = None  # 84 -> num_classes
    
    def forward(self, x):
        """
        Forward pass through LeNet-5.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 32, 32)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # TODO: Implement forward pass
        pass


class AlexNet:
    """
    AlexNet architecture for ImageNet classification.
    
    Revolutionary deep CNN that won ImageNet 2012, demonstrating the
    effectiveness of deep learning for computer vision.
    
    Key innovations:
    - ReLU activation
    - Dropout for regularization
    - Data augmentation
    - GPU implementation
    
    References:
        - Krizhevsky et al. (2012): "ImageNet Classification with Deep CNNs"
    """
    
    def __init__(self, num_classes=1000):
        """
        Initialize AlexNet.
        
        Args:
            num_classes: Number of output classes (default: 1000 for ImageNet)
        """
        # TODO: Implement AlexNet architecture
        # Conv layers: 96, 256, 384, 384, 256
        # FC layers: 4096, 4096, num_classes
        pass
    
    def forward(self, x):
        """
        Forward pass through AlexNet.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # TODO: Implement forward pass
        pass


class VGG:
    """
    VGG network architecture.
    
    Uses small 3x3 convolutions throughout, demonstrating that network
    depth is a critical component for good performance.
    
    Variants: VGG11, VGG13, VGG16, VGG19 (number indicates layers)
    
    References:
        - Simonyan & Zisserman (2014): "Very Deep CNNs for Large-Scale Image Recognition"
    """
    
    def __init__(self, config, num_classes=1000):
        """
        Initialize VGG network.
        
        Args:
            config: List specifying VGG architecture (e.g., [64, 'M', 128, 128, 'M', ...])
                   where numbers are conv channels and 'M' indicates max pooling
            num_classes: Number of output classes (default: 1000)
        """
        # TODO: Implement VGG architecture based on config
        pass
    
    def forward(self, x):
        """
        Forward pass through VGG.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # TODO: Implement forward pass
        pass


class ResNet:
    """
    Residual Network (ResNet) architecture.
    
    Introduces residual connections (skip connections) that enable training
    of very deep networks (100+ layers) by addressing the vanishing gradient
    problem.
    
    Variants: ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152
    
    References:
        - He et al. (2016): "Deep Residual Learning for Image Recognition"
    """
    
    def __init__(self, block_type, num_blocks, num_classes=1000):
        """
        Initialize ResNet.
        
        Args:
            block_type: Type of residual block ('basic' or 'bottleneck')
            num_blocks: List specifying number of blocks in each stage
            num_classes: Number of output classes (default: 1000)
        """
        # TODO: Implement ResNet architecture
        # Initial conv: 7x7, 64 channels
        # Stage 1: num_blocks[0] residual blocks, 64 channels
        # Stage 2: num_blocks[1] residual blocks, 128 channels
        # Stage 3: num_blocks[2] residual blocks, 256 channels
        # Stage 4: num_blocks[3] residual blocks, 512 channels
        # Global average pooling
        # FC layer
        pass
    
    def forward(self, x):
        """
        Forward pass through ResNet.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # TODO: Implement forward pass
        pass
