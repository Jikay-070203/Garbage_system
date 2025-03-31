class Concat(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.ultralytics.nn.modules.conv.___torch_mangle_154.Concat,
    argument_1: Tensor,
    argument_2: Tensor) -> Tensor:
    input = torch.cat([argument_1, argument_2], 1)
    return input
