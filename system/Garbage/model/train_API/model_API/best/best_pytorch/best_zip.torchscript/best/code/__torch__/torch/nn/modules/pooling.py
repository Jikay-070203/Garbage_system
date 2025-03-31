class MaxPool2d(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.torch.nn.modules.pooling.MaxPool2d,
    argument_1: Tensor) -> Tensor:
    input = torch.max_pool2d(argument_1, [5, 5], [1, 1], [2, 2], [1, 1])
    return input
  def forward1(self: __torch__.torch.nn.modules.pooling.MaxPool2d,
    argument_1: Tensor) -> Tensor:
    input = torch.max_pool2d(argument_1, [5, 5], [1, 1], [2, 2], [1, 1])
    return input
  def forward2(self: __torch__.torch.nn.modules.pooling.MaxPool2d,
    argument_1: Tensor) -> Tensor:
    _0 = torch.max_pool2d(argument_1, [5, 5], [1, 1], [2, 2], [1, 1])
    return _0
