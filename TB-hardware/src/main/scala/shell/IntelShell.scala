package shell


import chisel3._
import config._
import chipsalliance.rocketchip.config._


class IntelShell(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val host = new AXIClient(p(ShellKey).hostParams)
    val mem  = new AXIMaster(p(ShellKey).memParams)
  })

  val vcr = Module(new VCR)
  val vme = Module(new VME)
  // Connect the DNN core and its VME modules.
//   val core = Module(new DNNCore())

//   core.io.vcr <> vcr.io.vcr
//   vme.io.vme <> core.io.vme




  // For whatever reason; this is hoisted here wheras the xilinx shell includes a VTA shell separately. For the timebeing we retain it here.
  io.host.aw.ready := vcr.io.host.aw.ready
  vcr.io.host.aw.valid := io.host.aw.valid
  vcr.io.host.aw.bits.addr := io.host.aw.bits.addr
  io.host.w.ready := vcr.io.host.w.ready
  vcr.io.host.w.valid := io.host.w.valid
  vcr.io.host.w.bits.data := io.host.w.bits.data
  vcr.io.host.w.bits.strb := io.host.w.bits.strb
  vcr.io.host.b.ready := io.host.b.ready
  io.host.b.valid := vcr.io.host.b.valid
  io.host.b.bits.resp := vcr.io.host.b.bits.resp
  io.host.b.bits.id := io.host.w.bits.id

  io.host.ar.ready := vcr.io.host.ar.ready
  vcr.io.host.ar.valid := io.host.ar.valid
  vcr.io.host.ar.bits.addr := io.host.ar.bits.addr
  vcr.io.host.r.ready := io.host.r.ready
  io.host.r.valid := vcr.io.host.r.valid
  io.host.r.bits.data := vcr.io.host.r.bits.data
  io.host.r.bits.resp := vcr.io.host.r.bits.resp
  io.host.r.bits.id := io.host.ar.bits.id

  io.host.b.bits.user <> DontCare
  io.host.r.bits.user <> DontCare
  io.host.r.bits.last := 1.U

//  io.mem <> vme.io.mem
  io.mem <> DontCare
}

