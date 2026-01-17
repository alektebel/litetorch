"""
Test suite for neural network layers.
"""
import unittest


class TestLinearLayer(unittest.TestCase):
    """Test Linear (fully connected) layer."""
    
    def test_linear_forward(self):
        """Test forward pass of Linear layer."""
        # TODO: Implement after Linear layer is created
        pass
    
    def test_linear_backward(self):
        """Test backward pass of Linear layer."""
        # TODO: Implement after Linear layer is created
        pass
    
    def test_linear_output_shape(self):
        """Test output shape of Linear layer."""
        # TODO: Implement after Linear layer is created
        pass


class TestConv2d(unittest.TestCase):
    """Test Conv2d layer."""
    
    def test_conv2d_initialization(self):
        """Test Conv2d layer initialization."""
        from litetorch.nn import Conv2d
        
        # Test basic initialization
        conv = Conv2d(in_channels=3, out_channels=64, kernel_size=3, 
                     stride=1, padding=1)
        self.assertEqual(conv.in_channels, 3)
        self.assertEqual(conv.out_channels, 64)
        self.assertEqual(conv.kernel_size, (3, 3))
        self.assertEqual(conv.stride, 1)
        self.assertEqual(conv.padding, 1)
        
        # Check weight initialization
        self.assertIsNotNone(conv.weight)
        self.assertEqual(conv.weight.shape, (64, 3, 3, 3))
        
        # Check bias initialization
        self.assertTrue(conv.use_bias)
        self.assertIsNotNone(conv.bias)
        self.assertEqual(conv.bias.shape, (64,))
    
    def test_conv2d_forward(self):
        """Test forward pass of Conv2d layer."""
        # TODO: Implement after Conv2d forward pass is implemented
        pass
    
    def test_conv2d_backward(self):
        """Test backward pass of Conv2d layer."""
        # TODO: Implement after Conv2d backward pass is implemented
        pass
    
    def test_conv2d_output_shape(self):
        """Test output shape of Conv2d layer."""
        # TODO: Implement after Conv2d forward pass is implemented
        # Expected output shape: (batch, out_channels, out_height, out_width)
        # out_height = (height + 2*padding - kernel_size) / stride + 1
        pass


class TestMaxPool2d(unittest.TestCase):
    """Test MaxPool2d layer."""
    
    def test_maxpool2d_initialization(self):
        """Test MaxPool2d layer initialization."""
        from litetorch.nn import MaxPool2d
        
        # Test initialization with explicit stride
        pool = MaxPool2d(kernel_size=2, stride=2)
        self.assertEqual(pool.kernel_size, (2, 2))
        self.assertEqual(pool.stride, 2)
        
        # Test default stride (should equal kernel_size)
        pool2 = MaxPool2d(kernel_size=3)
        self.assertEqual(pool2.stride, (3, 3))
    
    def test_maxpool2d_forward(self):
        """Test forward pass of MaxPool2d layer."""
        # TODO: Implement after MaxPool2d forward pass is implemented
        pass
    
    def test_maxpool2d_backward(self):
        """Test backward pass of MaxPool2d layer."""
        # TODO: Implement after MaxPool2d backward pass is implemented
        pass


class TestBatchNorm2d(unittest.TestCase):
    """Test BatchNorm2d layer."""
    
    def test_batchnorm2d_initialization(self):
        """Test BatchNorm2d layer initialization."""
        from litetorch.nn import BatchNorm2d
        
        # Test initialization
        bn = BatchNorm2d(num_features=64)
        self.assertEqual(bn.num_features, 64)
        self.assertTrue(bn.affine)
        self.assertTrue(bn.training)
        
        # Check learnable parameters
        self.assertIsNotNone(bn.gamma)
        self.assertIsNotNone(bn.beta)
        self.assertEqual(bn.gamma.shape, (64,))
        self.assertEqual(bn.beta.shape, (64,))
        
        # Check running statistics
        self.assertEqual(bn.running_mean.shape, (64,))
        self.assertEqual(bn.running_var.shape, (64,))
    
    def test_batchnorm2d_forward(self):
        """Test forward pass of BatchNorm2d layer."""
        # TODO: Implement after BatchNorm2d forward pass is implemented
        pass
    
    def test_batchnorm2d_train_eval_modes(self):
        """Test train/eval mode switching."""
        from litetorch.nn import BatchNorm2d
        
        bn = BatchNorm2d(num_features=64)
        self.assertTrue(bn.training)
        
        bn.eval()
        self.assertFalse(bn.training)
        
        bn.train()
        self.assertTrue(bn.training)


class TestCNNBlocks(unittest.TestCase):
    """Test CNN building blocks."""
    
    def test_vgg_block_initialization(self):
        """Test VGGBlock initialization."""
        from litetorch.nn import VGGBlock
        
        # Test with 2 conv layers (default)
        block = VGGBlock(in_channels=64, out_channels=128, num_convs=2)
        self.assertEqual(len(block.convs), 2)
        self.assertIsNotNone(block.pool)
    
    def test_residual_block_initialization(self):
        """Test ResidualBlock initialization."""
        from litetorch.nn import ResidualBlock
        
        # Test basic residual block
        block = ResidualBlock(in_channels=64, out_channels=64, stride=1)
        self.assertIsNotNone(block.conv1)
        self.assertIsNotNone(block.conv2)
        self.assertIsNotNone(block.bn1)
        self.assertIsNotNone(block.bn2)
        
        # When channels/stride match, no shortcut needed
        self.assertIsNone(block.shortcut)
        
        # Test with stride=2 (should have shortcut)
        block2 = ResidualBlock(in_channels=64, out_channels=128, stride=2)
        self.assertIsNotNone(block2.shortcut)
        self.assertIsNotNone(block2.shortcut_bn)
    
    def test_bottleneck_block_initialization(self):
        """Test BottleneckBlock initialization."""
        from litetorch.nn import BottleneckBlock
        
        block = BottleneckBlock(in_channels=256, bottleneck_channels=64, 
                               out_channels=256, stride=1)
        self.assertIsNotNone(block.conv1)
        self.assertIsNotNone(block.conv2)
        self.assertIsNotNone(block.conv3)
    
    def test_inception_module_initialization(self):
        """Test InceptionModule initialization."""
        from litetorch.nn import InceptionModule
        
        module = InceptionModule(
            in_channels=192,
            ch1x1=64,
            ch3x3red=96, ch3x3=128,
            ch5x5red=16, ch5x5=32,
            pool_proj=32
        )
        self.assertIsNotNone(module.branch1)
        self.assertIsNotNone(module.branch2_1)
        self.assertIsNotNone(module.branch2_2)
    
    def test_depthwise_separable_conv_initialization(self):
        """Test DepthwiseSeparableConv2d initialization."""
        from litetorch.nn import DepthwiseSeparableConv2d
        
        conv = DepthwiseSeparableConv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.assertIsNotNone(conv.depthwise)
        self.assertIsNotNone(conv.pointwise)


class TestCNNArchitectures(unittest.TestCase):
    """Test complete CNN architectures."""
    
    def test_lenet5_initialization(self):
        """Test LeNet5 initialization."""
        from litetorch.nn import LeNet5
        
        model = LeNet5(num_classes=10)
        self.assertIsNotNone(model.conv1)
        self.assertIsNotNone(model.conv2)
        self.assertIsNotNone(model.pool1)
        self.assertIsNotNone(model.pool2)
    
    def test_alexnet_initialization(self):
        """Test AlexNet initialization."""
        # TODO: Implement after AlexNet is fully implemented
        pass
    
    def test_vgg_initialization(self):
        """Test VGG initialization."""
        # TODO: Implement after VGG is fully implemented
        pass
    
    def test_resnet_initialization(self):
        """Test ResNet initialization."""
        # TODO: Implement after ResNet is fully implemented
        pass


class TestActivations(unittest.TestCase):
    """Test activation functions."""
    
    def test_relu_forward(self):
        """Test ReLU forward pass."""
        # TODO: Implement after ReLU is created
        pass
    
    def test_sigmoid_forward(self):
        """Test Sigmoid forward pass."""
        # TODO: Implement after Sigmoid is created
        pass
    
    def test_tanh_forward(self):
        """Test Tanh forward pass."""
        # TODO: Implement after Tanh is created
        pass
    
    def test_softmax_forward(self):
        """Test Softmax forward pass."""
        # TODO: Implement after Softmax is created
        pass


class TestLossFunctions(unittest.TestCase):
    """Test loss functions."""
    
    def test_mse_loss(self):
        """Test Mean Squared Error loss."""
        # TODO: Implement after MSELoss is created
        pass
    
    def test_cross_entropy_loss(self):
        """Test Cross Entropy loss."""
        # TODO: Implement after CrossEntropyLoss is created
        pass


if __name__ == '__main__':
    unittest.main()
