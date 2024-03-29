package accel

/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

import FPU.FType
import arbiters.TypeStackFile
import chisel3.{when, _}
import chisel3.util._
import config._
import chipsalliance.rocketchip.config._
import chipsalliance.rocketchip.config._
import control.BasicBlockNoMaskNode
import dnn.memory.{ReadTensorController, TensorLoad, TensorMaster, TensorStore, WriteTensorController, inDMA_act, inDMA_wgt, outDMA_act}
import interfaces.{ControlBundle, CustomDataBundle, DataBundle}
import junctions.SplitCallNew
import node.{FPvecN, FXmatNxN, FXvecN, UnTypStore, matNxN, vecN}
import shell._
import dnn.memory.ISA._
import dnn_layers.{DW_Block, DW_PW_Block, PDP_Block, PW_Block}
import dnnnode.{Mac2dTensor, ShapeTransformer, StoreQueue, TLoad, TStore, WeightShapeTransformer}
import firrtl.transforms.DontTouchAnnotation


/** DNNCore.
  *
  * The DNNcore defines the current DNN accelerator by connecting the M-Bricks, L-Bricks and I-Bricks together.
  * By changing the parameters and batch size, the bricks will be configured automatically.
  */
class DNNCore(Hx: Int = 3, Wx: Int = 3, Cx: Int = 2, Cb: Int = 2, Fx: Int = 1, Px: Int = 1, K: Int = 3,
              val num_ptrs: Int = 3, val num_vals: Int = 2, val num_event: Int = 4, val num_ctrl: Int = 1) (implicit val p: Parameters) extends Module {
  val io = IO(new Bundle {
    val vcr = new VCRClient
    val vme = new VMEMaster
  })

  val cycle_count = new Counter(2000)

  val S = new FType(8, 24)

  //  val memShape = new vecN(16, 0, false)
  val memShape = new FPvecN(16, S, 0)
  val CxShape = new FPvecN(Cx, S, 0)

  val CxShape2 = new FXvecN(Cx, fraction = 10, 0)

  val conv = Module(new PW_Block(Hx, Fx, Cb, "intWgtPW1", "inp")(memShape)(CxShape))

  /* ================================================================== *
     *                      Basic Block signals                         *
     * ================================================================== */
  conv.io.wgtIndex := 0.U

  conv.io.rowWidth := Wx.U //3.U

  /* ================================================================== *
     *                           Connections                            *
     * ================================================================== */

  io.vcr.ecnt(0).bits := cycle_count.value

  io.vcr.ecnt(1).bits := conv.io.inDMA_act_time
  io.vcr.ecnt(2).bits := conv.io.inDMA_wgt_time
  io.vcr.ecnt(3).bits := conv.io.mac_time

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

