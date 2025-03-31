class C2f(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  cv1 : __torch__.ultralytics.nn.modules.conv.___torch_mangle_66.Conv
  cv2 : __torch__.ultralytics.nn.modules.conv.___torch_mangle_69.Conv
  m : __torch__.torch.nn.modules.container.___torch_mangle_98.ModuleList
  def forward(self: __torch__.ultralytics.nn.modules.block.___torch_mangle_99.C2f,
    argument_1: __torch__.torch.nn.modules.activation.___torch_mangle_273.SiLU,
    argument_2: Tensor) -> Tensor:
    cv2 = self.cv2
    m = self.m
    _3 = getattr(m, "3")
    m0 = self.m
    _2 = getattr(m0, "2")
    m1 = self.m
    _1 = getattr(m1, "1")
    m2 = self.m
    _0 = getattr(m2, "0")
    cv1 = self.cv1
    _4 = (cv1).forward(argument_1, argument_2, )
    _5 = torch.split_with_sizes(_4, [192, 192], 1)
    _6, input, = _5
    _7 = (_0).forward(argument_1, input, )
    _8 = (_1).forward(argument_1, _7, )
    _9 = (_2).forward(argument_1, _8, )
    _10 = [_6, input, _7, _8, _9, (_3).forward(argument_1, _9, )]
    input0 = torch.cat(_10, 1)
    return (cv2).forward(argument_1, input0, )
