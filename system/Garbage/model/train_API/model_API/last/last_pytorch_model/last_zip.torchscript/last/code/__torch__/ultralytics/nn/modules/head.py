class Detect(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  cv2 : __torch__.torch.nn.modules.container.___torch_mangle_252.ModuleList
  cv3 : __torch__.torch.nn.modules.container.___torch_mangle_277.ModuleList
  dfl : __torch__.ultralytics.nn.modules.block.DFL
  def forward(self: __torch__.ultralytics.nn.modules.head.Detect,
    argument_1: Tensor,
    argument_2: Tensor,
    argument_3: Tensor) -> Tensor:
    dfl = self.dfl
    cv3 = self.cv3
    _2 = getattr(cv3, "2")
    cv2 = self.cv2
    _20 = getattr(cv2, "2")
    cv30 = self.cv3
    _1 = getattr(cv30, "1")
    cv20 = self.cv2
    _10 = getattr(cv20, "1")
    cv31 = self.cv3
    _0 = getattr(cv31, "0")
    cv32 = self.cv3
    _21 = getattr(cv32, "2")
    _11 = getattr(_21, "1")
    act = _11.act
    cv21 = self.cv2
    _00 = getattr(cv21, "0")
    _3 = [(_00).forward(act, argument_1, ), (_0).forward(act, argument_1, )]
    xi = torch.cat(_3, 1)
    _4 = [(_10).forward(act, argument_2, ), (_1).forward(act, argument_2, )]
    xi0 = torch.cat(_4, 1)
    _5 = [(_20).forward(act, argument_3, ), (_2).forward(argument_3, )]
    xi1 = torch.cat(_5, 1)
    _6 = ops.prim.NumToTensor(torch.size(xi, 0))
    _7 = int(_6)
    _8 = int(_6)
    _9 = [torch.view(xi, [int(_6), 74, -1]), torch.view(xi0, [_8, 74, -1]), torch.view(xi1, [_7, 74, -1])]
    _12 = torch.split_with_sizes(torch.cat(_9, 2), [64, 10], 1)
    x, cls, = _12
    _13 = (dfl).forward(x, )
    anchor_points = torch.unsqueeze(CONSTANTS.c0, 0)
    lt, rb, = torch.chunk(_13, 2, 1)
    x1y1 = torch.sub(anchor_points, lt)
    x2y2 = torch.add(anchor_points, rb)
    c_xy = torch.div(torch.add(x1y1, x2y2), CONSTANTS.c1)
    wh = torch.sub(x2y2, x1y1)
    dbox = torch.mul(torch.cat([c_xy, wh], 1), CONSTANTS.c2)
    _14 = torch.cat([dbox, torch.sigmoid(cls)], 1)
    return _14
