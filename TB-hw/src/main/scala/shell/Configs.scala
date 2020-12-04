package shell


import chipsalliance.rocketchip.config._


class VCRSimParams(val num_ptrs: Int = 4, val num_vals: Int = 2,
                   val num_event: Int = 1, val num_ctrl: Int = 1) extends VCRParams {
  override val nCtrl = num_ctrl
  override val nECnt = num_event
  override val nVals = num_vals
  override val nPtrs = num_ptrs
  override val regBits = 32
}


/** VME parameters.
  *
  * These parameters are used on VME interfaces and modules.
  */
class VMESimParams(CONV: String = "PW", Hx: Int = 3, Wx: Int = 3, Cx: Int = 2, Cb: Int = 2, Fx: Int = 1, Px: Int = 1, K: Int = 3) extends VMEParams {
//  override val nReadClients: Int  = Hx + 1
//  override val nWriteClients: Int = Hx * Fx
  var nRead = 1
  var nWrite = 1
  if(CONV == "PW") {
    nRead = Hx + 1
    nWrite = Hx * Fx
  } else if (CONV == "PDP") {
    nRead = Hx + 3
    nWrite = Px * (Hx - K + 1)
  }

  override val nReadClients: Int  = nRead
  override val nWriteClients: Int = nWrite

  require(nReadClients > 0,
    s"\n\n[Dandelion] [VMEParams] nReadClients must be larger than 0\n\n")
  require(
    nWriteClients > 0,
    s"\n\n[Dandelion] [VMEParams] nWriteClients must be larger than 0\n\n")
}


/**
  * PW Configs:
  * nReadClients = Hx + 1
  * nwriteClients = Hx * Fx
  */

/** De10Config. Shell configuration for De10 */
class De10Config(CONV: String = "PW", Hx: Int = 3, Wx: Int = 3, Cx: Int = 2, Cb: Int = 2, Fx: Int = 1, Px: Int = 1, K: Int = 3,
                 val num_ptrs: Int = 3, val num_vals: Int = 2, val num_event: Int = 4, val num_ctrl: Int = 1) extends Config((site, here, up) => {
  case ShellKey => ShellParams(
    hostParams = AXIParams(
      addrBits = 16, dataBits = 32, idBits = 13, lenBits = 4),
    memParams = AXIParams(
      addrBits = 32, dataBits = 64, userBits = 5,
      lenBits = 4, // limit to 16 beats, instead of 256 beats in AXI4
      coherent = true),
    vcrParams = new VCRSimParams(num_ptrs, num_vals, num_event, num_ctrl),
    vmeParams = new VMESimParams(CONV = CONV, Hx = Hx, Wx = Wx, Cx = Cx, Cb = Cb, Fx = Fx, Px = Px, K = K))
})


/** PynqConfig. Shell configuration for Pynq */
class PynqConfig(CONV: String = "PW", Hx: Int = 3, Wx: Int = 3, Cx: Int = 2, Cb: Int = 2, Fx: Int = 1, Px: Int = 1, K: Int = 3,
                 val num_ptrs: Int = 3, val num_vals: Int = 2, val num_event: Int = 4, val num_ctrl: Int = 1) extends Config((site, here, up) => {
  case ShellKey => ShellParams(
    hostParams = AXIParams(
      coherent = false,
      addrBits = 16,
      dataBits = 32,
      lenBits = 8,
      userBits = 1),
    memParams = AXIParams(
      coherent = true,
      addrBits = 32,
      dataBits = 64,
      lenBits = 8,
      userBits = 1),
    vcrParams = new VCRSimParams(num_ptrs, num_vals, num_event, num_ctrl),
    vmeParams = new VMESimParams(CONV = CONV, Hx = Hx, Wx = Wx, Cx = Cx, Cb = Cb, Fx = Fx, Px = Px, K = K))
})


/** De10Config. Shell configuration for De10 */
class F1Config(CONV: String = "PW", Hx: Int = 3, Wx: Int = 3, Cx: Int = 2, Cb: Int = 2, Fx: Int = 1, Px: Int = 1, K: Int = 3,
               val num_ptrs: Int = 3, val num_vals: Int = 2, val num_event: Int = 4, val num_ctrl: Int = 1)
  extends Config((site, here, up) => {
    case ShellKey => ShellParams(
      hostParams = AXIParams(
        addrBits = 32, dataBits = 32, idBits = 13, lenBits = 8),
      memParams = AXIParams(
        addrBits = 64, dataBits = 512, userBits = 10,
        lenBits = 8, // limit to 16 beats, instead of 256 beats in AXI4
        coherent = false),
      vcrParams = new VCRSimParams(num_ptrs, num_vals, num_event, num_ctrl),
      vmeParams = new VMESimParams(CONV = CONV, Hx = Hx, Wx = Wx, Cx = Cx, Cb = Cb, Fx = Fx, Px = Px, K = K))
  })

