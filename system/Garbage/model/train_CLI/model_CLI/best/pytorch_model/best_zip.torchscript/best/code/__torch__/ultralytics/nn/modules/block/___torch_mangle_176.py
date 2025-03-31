class C2f(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  cv1 : __torch__.ultralytics.nn.modules.conv.___torch_mangle_157.Conv
  cv2 : __torch__.ultralytics.nn.modules.conv.___torch_mangle_160.Conv
  m : __torch__.torch.nn.modules.container.___torch_mangle_175.ModuleList
  def forward(self: __torch__.ultralytics.nn.modules.block.___torch_mangle_176.C2f,
    argument_1: __torch__.torch.nn.modules.activation.___torch_mangle_273.SiLU,
    argument_2: Tensor) -> Tensor:
    cv2 = self.cv2
    m = self.m
    _1 = getattr(m, "1")
    m0 = self.m
    _0 = getattr(m0, "0")
    cv1 = self.cv1
    _2 = (cv1).forward(argument_1, argument_2, )
    _3 = torch.split_with_sizes(_2, [96, 96], 1)
    _4, input, = _3
    _5 = (_0).forward(argument_1, input, )
    _6 = [_4, input, _5, (_1).forward(argument_1, _5, )]
    input0 = torch.cat(_6, 1)
    return (cv2).forward(argument_1, input0, )
