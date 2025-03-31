class Bottleneck(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  cv1 : __torch__.ultralytics.nn.modules.conv.___torch_mangle_54.Conv
  cv2 : __torch__.ultralytics.nn.modules.conv.___torch_mangle_57.Conv
  def forward(self: __torch__.ultralytics.nn.modules.block.___torch_mangle_58.Bottleneck,
    argument_1: __torch__.torch.nn.modules.activation.___torch_mangle_273.SiLU,
    argument_2: Tensor) -> Tensor:
    cv2 = self.cv2
    cv1 = self.cv1
    _0 = (cv1).forward(argument_1, argument_2, )
    _1 = torch.add(argument_2, (cv2).forward(argument_1, _0, ))
    return _1
