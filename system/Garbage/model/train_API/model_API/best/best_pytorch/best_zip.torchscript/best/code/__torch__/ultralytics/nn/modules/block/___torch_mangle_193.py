class Bottleneck(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  cv1 : __torch__.ultralytics.nn.modules.conv.___torch_mangle_189.Conv
  cv2 : __torch__.ultralytics.nn.modules.conv.___torch_mangle_192.Conv
  def forward(self: __torch__.ultralytics.nn.modules.block.___torch_mangle_193.Bottleneck,
    argument_1: __torch__.torch.nn.modules.activation.___torch_mangle_273.SiLU,
    input: Tensor) -> Tensor:
    cv2 = self.cv2
    cv1 = self.cv1
    _0 = (cv2).forward(argument_1, (cv1).forward(argument_1, input, ), )
    return _0
