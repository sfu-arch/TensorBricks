package shell


import chisel3._
import chipsalliance.rocketchip.config._
import accel._
import dandelion.shell.ConfigBusMaster


/** F1Shell.
  *
  * This is a wrapper shell mostly used to match F1-Xilinx convention naming,
  * therefore we can pack VTA as an IP for IPI based flows.
  */
class F1Shell(CONV: String = "PW", Hx: Int = 3, Wx: Int = 3, Cx: Int = 2, Cb: Int = 2, Fx: Int = 1, Px: Int = 1, K: Int = 3,
              numPtrs: Int, numVals: Int, numRets: Int, numEvents: Int, numCtrls: Int)(implicit p: Parameters) extends RawModule {

  val hp = p(ShellKey).hostParams
  val mp = p(ShellKey).memParams

  val ap_clk = IO(Input(Clock()))
  val ap_rst_n = IO(Input(Bool()))
  val cl_axi_mstr_bus = IO(new XilinxAXIMaster(mp))
  val axi_mstr_cfg_bus = IO(new ConfigBusMaster(hp))

  val shell = withClockAndReset(clock = ap_clk, reset = ~ap_rst_n) {
    Module(new DNNF1Accel(CONV = CONV, Hx = Hx, Wx = Wx, Cx = Cx, Cb = Cb, Fx = Fx, Px = Px, K = K,
      num_ptrs = numPtrs, num_vals = numVals, num_event = numEvents, num_ctrl = numCtrls))
  }


  // memory
  cl_axi_mstr_bus.AWVALID := shell.io.mem.aw.valid
  shell.io.mem.aw.ready := cl_axi_mstr_bus.AWREADY
  cl_axi_mstr_bus.AWADDR := shell.io.mem.aw.bits.addr
  cl_axi_mstr_bus.AWID := shell.io.mem.aw.bits.id
  cl_axi_mstr_bus.AWUSER := shell.io.mem.aw.bits.user
  cl_axi_mstr_bus.AWLEN := shell.io.mem.aw.bits.len
  cl_axi_mstr_bus.AWSIZE := shell.io.mem.aw.bits.size
  cl_axi_mstr_bus.AWBURST := shell.io.mem.aw.bits.burst
  cl_axi_mstr_bus.AWLOCK := shell.io.mem.aw.bits.lock
  cl_axi_mstr_bus.AWCACHE := shell.io.mem.aw.bits.cache
  cl_axi_mstr_bus.AWPROT := shell.io.mem.aw.bits.prot
  cl_axi_mstr_bus.AWQOS := shell.io.mem.aw.bits.qos
  cl_axi_mstr_bus.AWREGION := shell.io.mem.aw.bits.region

  cl_axi_mstr_bus.WVALID := shell.io.mem.w.valid
  shell.io.mem.w.ready := cl_axi_mstr_bus.WREADY
  cl_axi_mstr_bus.WDATA := shell.io.mem.w.bits.data
  cl_axi_mstr_bus.WSTRB := shell.io.mem.w.bits.strb
  cl_axi_mstr_bus.WLAST := shell.io.mem.w.bits.last
  cl_axi_mstr_bus.WID := shell.io.mem.w.bits.id
  cl_axi_mstr_bus.WUSER := shell.io.mem.w.bits.user

  shell.io.mem.b.valid := cl_axi_mstr_bus.BVALID
  cl_axi_mstr_bus.BREADY := shell.io.mem.b.valid
  shell.io.mem.b.bits.resp := cl_axi_mstr_bus.BRESP
  shell.io.mem.b.bits.id := cl_axi_mstr_bus.BID
  shell.io.mem.b.bits.user := cl_axi_mstr_bus.BUSER

  cl_axi_mstr_bus.ARVALID := shell.io.mem.ar.valid
  shell.io.mem.ar.ready := cl_axi_mstr_bus.ARREADY
  cl_axi_mstr_bus.ARADDR := shell.io.mem.ar.bits.addr
  cl_axi_mstr_bus.ARID := shell.io.mem.ar.bits.id
  cl_axi_mstr_bus.ARUSER := shell.io.mem.ar.bits.user
  cl_axi_mstr_bus.ARLEN := shell.io.mem.ar.bits.len
  cl_axi_mstr_bus.ARSIZE := shell.io.mem.ar.bits.size
  cl_axi_mstr_bus.ARBURST := shell.io.mem.ar.bits.burst
  cl_axi_mstr_bus.ARLOCK := shell.io.mem.ar.bits.lock
  cl_axi_mstr_bus.ARCACHE := shell.io.mem.ar.bits.cache
  cl_axi_mstr_bus.ARPROT := shell.io.mem.ar.bits.prot
  cl_axi_mstr_bus.ARQOS := shell.io.mem.ar.bits.qos
  cl_axi_mstr_bus.ARREGION := shell.io.mem.ar.bits.region

  shell.io.mem.r.valid := cl_axi_mstr_bus.RVALID
  cl_axi_mstr_bus.RREADY := shell.io.mem.r.ready
  shell.io.mem.r.bits.data := cl_axi_mstr_bus.RDATA
  shell.io.mem.r.bits.resp := cl_axi_mstr_bus.RRESP
  shell.io.mem.r.bits.last := cl_axi_mstr_bus.RLAST
  shell.io.mem.r.bits.id := cl_axi_mstr_bus.RID
  shell.io.mem.r.bits.user := cl_axi_mstr_bus.RUSER

  // host
  shell.io.host.addr := axi_mstr_cfg_bus.addr
  shell.io.host.wdata := axi_mstr_cfg_bus.wdata
  shell.io.host.wr := axi_mstr_cfg_bus.wr
  shell.io.host.rd := axi_mstr_cfg_bus.rd
  axi_mstr_cfg_bus.ack := shell.io.host.ack
  axi_mstr_cfg_bus.rdata := shell.io.host.rdata

}
