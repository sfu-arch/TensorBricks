package accel

import FPU.FType
import chisel3.util._
import chisel3.{when, _}
import config._
import chipsalliance.rocketchip.config._
import chipsalliance.rocketchip.config._
import dnn_layers.{PDP_Block, PW_Block}
import node.{FPvecN, matNxN, vecN}
import shell._

/** DNNCore.
  *
  * The DNNcore defines the current DNN accelerator by connecting the M-Bricks, L-Bricks and I-Bricks together.
  * By changing the parameters and batch size, the bricks will be configured automatically.
  */
class DNNCorePW(Hx: Int = 3, Wx: Int = 3, Cx: Int = 2, Cb: Int = 2, Fx: Int = 1, Px: Int = 1, K: Int = 3,
                val num_ptrs: Int = 3, val num_vals: Int = 2, val num_event: Int = 4, val num_ctrl: Int = 1)(implicit val p: Parameters) extends Module {
  require(num_ptrs > 2, "num_ptrs should be at least 3")

  val io = IO(new Bundle {
    val vcr = new VCRClient
    val vme = new VMEMaster
  })

  val cycle_count = new Counter(2000)

  val S = new FType(8, 24)

  val memShape = new vecN(16, 0, false)
//  val memShape = new FPvecN(16, S, 0)

  val macDWShape = new matNxN(K, false)
  val macPW2Shape = new vecN(Fx, 0, false)

  val wgtDWShape = new vecN(K * K, 0, false)
  val wgtPW2Shape = new vecN(Fx, 0, false)

  val CxShape = new vecN(Cx, 0, false)
//  val CxShape = new FPvecN(Cx, S, 0)

//  val DW_B1 = Module(new DW_Block(3, "wgt", "inp")(memShape)(wgtDWShape)(macDWShape))

  val conv = Module(new PW_Block(Hx, Fx, Cb, "intWgtPW1", "inp")(memShape)(CxShape))
//  val conv = Module(new PDP_Block(Hx, K, Fx, 19, Px,
//                    "intWgtPW1", "intWgtDW", "intWgtPW2", "inp")
//                    (memShape)(CxShape)
//                    (wgtDWShape)(macDWShape)
//                    (wgtPW2Shape)(macPW2Shape))

  /* ================================================================== *
     *                      Basic Block signals                         *
     * ================================================================== */
  conv.io.wgtIndex := 0.U

  conv.io.rowWidth := Wx.U //3.U

  /* ================================================================== *
     *                           Connections                            *
     * ================================================================== */

  io.vcr.ecnt(0).bits := cycle_count.value

  if (num_event > 1){
    for (i <- 1 until num_event){
      io.vcr.ecnt(i).bits := conv.io.inDMA_act_time
    }
  }

  /* ================================================================== *
    *                    VME Reads and writes                           *
    * ================================================================== */

  for (i <- 0 until Hx) {
    io.vme.rd(i) <> conv.io.vme_rd(i)
  }
  io.vme.rd(Hx) <> conv.io.vme_wgt_rd

  for (i <- 0 until Fx * Hx) {
    io.vme.wr(i) <> conv.io.vme_wr(i)
  }


  conv.io.start := false.B

  conv.io.inBaseAddr := io.vcr.ptrs(0)

  conv.io.wgt_baddr := io.vcr.ptrs(1)

  conv.io.outBaseAddr := io.vcr.ptrs(2)

  val sIdle :: sExec :: sFinish :: Nil = Enum(3)

  val state = RegInit(sIdle)
  switch(state) {
    is(sIdle) {
      when(io.vcr.launch) {
        conv.io.start := true.B
        state := sExec
      }
    }
    is(sExec) {
      when(conv.io.done) {
        state := sIdle
      }
    }
  }

  val last = state === sExec && conv.io.done
  io.vcr.finish := last
  io.vcr.ecnt(0).valid := last
  io.vcr.ecnt(1).valid := last
  io.vcr.ecnt(2).valid := last
  io.vcr.ecnt(3).valid := last

  when(state =/= sIdle) {
    cycle_count.inc()
  }
}
