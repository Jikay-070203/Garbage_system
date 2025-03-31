class Conv2d(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.torch.nn.modules.conv.___torch_mangle_184.Conv2d,
    input: Tensor) -> Tensor:
    bias = self.bias
    weight = self.weight
    input0 = torch._convolution(input, weight, bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1, False, False, True, True)
    return input0
