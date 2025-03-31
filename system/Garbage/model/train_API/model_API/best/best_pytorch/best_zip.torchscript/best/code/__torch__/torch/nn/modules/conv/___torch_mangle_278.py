class Conv2d(Module):
  __parameters__ = ["weight", ]
  __buffers__ = []
  weight : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.torch.nn.modules.conv.___torch_mangle_278.Conv2d,
    input: Tensor) -> Tensor:
    weight = self.weight
    _0 = torch._convolution(input, weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1, False, False, True, True)
    return _0
