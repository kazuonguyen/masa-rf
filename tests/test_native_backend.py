"""Operator checks use optional GPU environment, not the lightweight CLI env."""
import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("mmcv")
from era_masa.native_masa import TorchvisionDCNv2


def test_zero_offset_dcn_equals_masked_convolution():
    torch.manual_seed(7)
    layer = TorchvisionDCNv2(2, 3, 3, padding=1)
    with torch.no_grad():
        layer.weight.normal_()
        layer.bias.normal_()
    image = torch.randn(1, 2, 6, 7)
    offset = torch.zeros(1, 18, 6, 7)
    mask = torch.full((1, 9, 6, 7), 0.4)
    result = layer(image, offset, mask)
    expected = torch.nn.functional.conv2d(image, layer.weight * 0.4, layer.bias, padding=1)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_nonzero_dcn_cpu_cuda_agree():
    torch.manual_seed(9)
    layer = TorchvisionDCNv2(2, 3, 3, padding=1)
    with torch.no_grad():
        layer.weight.normal_()
        layer.bias.normal_()
    image = torch.randn(1, 2, 6, 7)
    offset = torch.randn(1, 18, 6, 7) * 0.5
    mask = torch.rand(1, 9, 6, 7)
    expected = layer(image, offset, mask)
    actual = layer.cuda()(image.cuda(), offset.cuda(), mask.cuda()).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
