package shell


import chisel3._
import chipsalliance.rocketchip.config._
import accel._


/** XilinxShell.
  *
  * This is a wrapper shell mostly used to match Xilinx convention naming,
  * therefore we can pack VTA as an IP for IPI based flows.
  */
class XilinxShell(implicit p: Parameters) extends RawModule {

  val hp = p(ShellKey).hostParams
  val mp = p(ShellKey).memParams

  val ap_clk = IO(Input(Clock()))
  val ap_rst_n = IO(Input(Bool()))
  val m_axi_gmem = IO(new XilinxAXIMaster(mp))
  val s_axi_control = IO(new XilinxAXILiteClient(hp))

  val shell = withClockAndReset (clock = ap_clk, reset = ~ap_rst_n) { Module(new DNNAccel()) }

  // memory
  m_axi_gmem.AWVALID := shell.io.mem.aw.valid
  shell.io.mem.aw.ready := m_axi_gmem.AWREADY
  m_axi_gmem.AWADDR := shell.io.mem.aw.bits.addr
  m_axi_gmem.AWID := shell.io.mem.aw.bits.id
  m_axi_gmem.AWUSER := shell.io.mem.aw.bits.user
  m_axi_gmem.AWLEN := shell.io.mem.aw.bits.len
  m_axi_gmem.AWSIZE := shell.io.mem.aw.bits.size
  m_axi_gmem.AWBURST := shell.io.mem.aw.bits.burst
  m_axi_gmem.AWLOCK := shell.io.mem.aw.bits.lock
  m_axi_gmem.AWCACHE := shell.io.mem.aw.bits.cache
  m_axi_gmem.AWPROT := shell.io.mem.aw.bits.prot
  m_axi_gmem.AWQOS := shell.io.mem.aw.bits.qos
  m_axi_gmem.AWREGION := shell.io.mem.aw.bits.region
  m_axi_gmem.WID <> DontCare

  m_axi_gmem.WVALID := shell.io.mem.w.valid
  shell.io.mem.w.ready := m_axi_gmem.WREADY
  m_axi_gmem.WDATA := shell.io.mem.w.bits.data
  m_axi_gmem.WSTRB := shell.io.mem.w.bits.strb
  m_axi_gmem.WLAST := shell.io.mem.w.bits.last
  //m_axi_gmem.WID := shell.io.mem.w.bits.id
  m_axi_gmem.WUSER := shell.io.mem.w.bits.user

  shell.io.mem.b.valid := m_axi_gmem.BVALID
  m_axi_gmem.BREADY := shell.io.mem.b.valid
  shell.io.mem.b.bits.resp := m_axi_gmem.BRESP
  shell.io.mem.b.bits.id := m_axi_gmem.BID
  shell.io.mem.b.bits.user := m_axi_gmem.BUSER

  m_axi_gmem.ARVALID := shell.io.mem.ar.valid
  shell.io.mem.ar.ready := m_axi_gmem.ARREADY
  m_axi_gmem.ARADDR := shell.io.mem.ar.bits.addr
  m_axi_gmem.ARID := shell.io.mem.ar.bits.id
  m_axi_gmem.ARUSER := shell.io.mem.ar.bits.user
  m_axi_gmem.ARLEN := shell.io.mem.ar.bits.len
  m_axi_gmem.ARSIZE := shell.io.mem.ar.bits.size
  m_axi_gmem.ARBURST := shell.io.mem.ar.bits.burst
  m_axi_gmem.ARLOCK := shell.io.mem.ar.bits.lock
  m_axi_gmem.ARCACHE := shell.io.mem.ar.bits.cache
  m_axi_gmem.ARPROT := shell.io.mem.ar.bits.prot
  m_axi_gmem.ARQOS := shell.io.mem.ar.bits.qos
  m_axi_gmem.ARREGION := shell.io.mem.ar.bits.region

  shell.io.mem.r.valid := m_axi_gmem.RVALID
  m_axi_gmem.RREADY := shell.io.mem.r.ready
  shell.io.mem.r.bits.data := m_axi_gmem.RDATA
  shell.io.mem.r.bits.resp := m_axi_gmem.RRESP
  shell.io.mem.r.bits.last := m_axi_gmem.RLAST
  shell.io.mem.r.bits.id := m_axi_gmem.RID
  shell.io.mem.r.bits.user := m_axi_gmem.RUSER

  // host
  shell.io.host.aw.valid := s_axi_control.AWVALID
  s_axi_control.AWREADY := shell.io.host.aw.ready
  shell.io.host.aw.bits.addr := s_axi_control.AWADDR

  shell.io.host.w.valid := s_axi_control.WVALID
  s_axi_control.WREADY := shell.io.host.w.ready
  shell.io.host.w.bits.data := s_axi_control.WDATA
  shell.io.host.w.bits.strb := s_axi_control.WSTRB

  s_axi_control.BVALID := shell.io.host.b.valid
  shell.io.host.b.ready := s_axi_control.BREADY
  s_axi_control.BRESP := shell.io.host.b.bits.resp

  shell.io.host.ar.valid := s_axi_control.ARVALID
  s_axi_control.ARREADY := shell.io.host.ar.ready
  shell.io.host.ar.bits.addr := s_axi_control.ARADDR

  s_axi_control.RVALID := shell.io.host.r.valid
  shell.io.host.r.ready := s_axi_control.RREADY
  s_axi_control.RDATA := shell.io.host.r.bits.data
  s_axi_control.RRESP := shell.io.host.r.bits.resp
}
