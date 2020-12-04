package accel

import accel.TensorBricksSimAccelMain.{CONV, num_ctrls, num_events, num_ptrs, num_returns, num_vals}
import chisel3._
import chisel3.MultiIOModule
import shell._
import vta.shell._
import shell.De10Config
import config._
import chipsalliance.rocketchip.config._
import dnn.memory._

class TensorBricksSimAccel(CONV: String = "PW", Hx: Int = 3, Wx: Int = 3, Cx: Int = 2, Cb: Int = 2, Fx: Int = 1, Px: Int = 1, K: Int = 3,
                           numPtrs: Int, numVals: Int, numRets: Int, numEvents: Int, numCtrls: Int)
                          (implicit val p: Parameters) extends MultiIOModule {
  val sim_clock = IO(Input(Clock()))
  val sim_wait = IO(Output(Bool()))
  val sim_shell = Module(new AXISimShell)
  val vta_shell = Module(new DNNAccel(CONV = CONV, Hx = Hx, Wx = Wx, Cx = Cx, Cb = Cb, Fx = Fx, Px = Px, K = K,
    num_ptrs = numPtrs, num_vals = numVals, num_event = numEvents, num_ctrl = numCtrls))
  sim_shell.sim_clock := sim_clock
  sim_wait := sim_shell.sim_wait

  sim_shell.mem.ar <> vta_shell.io.mem.ar
  sim_shell.mem.aw <> vta_shell.io.mem.aw
  vta_shell.io.mem.r <> sim_shell.mem.r
  vta_shell.io.mem.b <> sim_shell.mem.b
  sim_shell.mem.w <> vta_shell.io.mem.w


  vta_shell.io.host.ar <> sim_shell.host.ar
  vta_shell.io.host.aw <> sim_shell.host.aw
  sim_shell.host.r <> vta_shell.io.host.r
  sim_shell.host.b <> vta_shell.io.host.b
  vta_shell.io.host.w <> sim_shell.host.w

}


class TensorBricksDe10Configs(CONV: String = "PW", Hx: Int = 3, Wx: Int = 3, Cx: Int = 2, Cb: Int = 2, Fx: Int = 1, Px: Int = 1, K: Int = 3,
                              numPtrs: Int, numVals: Int, numRets: Int, numEvents: Int, numCtrls: Int)
  extends Config(new De10Config(CONV = CONV, Hx = Hx, Wx = Wx, Cx = Cx, Cb = Cb, Fx = Fx, Px = Px, K = K,
    num_ptrs = numPtrs, num_vals = numVals, num_event = numEvents, num_ctrl = numCtrls) ++
    new CoreConfig ++ new WithAccelConfig)

class TensorBricksF1Configs(CONV: String = "PW", Hx: Int = 3, Wx: Int = 3, Cx: Int = 2, Cb: Int = 2, Fx: Int = 1, Px: Int = 1, K: Int = 3,
                            numPtrs: Int, numVals: Int, numRets: Int, numEvents: Int, numCtrls: Int)
  extends Config(new F1Config(CONV = CONV, Hx = Hx, Wx = Wx, Cx = Cx, Cb = Cb, Fx = Fx, Px = Px, K = K,
    num_ptrs = numPtrs, num_vals = numVals, num_event = numEvents, num_ctrl = numCtrls) ++
    new CoreConfig(
      CoreParams(
        batch = 1,
        blockOut = 16,
        blockIn = 16, //16,
        inpBits = 32,//8,
        wgtBits = 32,//8,
        kernelSize = 9,
        PW1kernelSize = 5,
        PW2kernelSize = 5,
        uopBits = 32,
        accBits = 32,
        outBits = 32,
        uopMemDepth = 2048,
        inpMemDepth = 200, //2660
        wgtMemDepth = 512,

        extWgtP1MemDepth = 475,
        extWgtDMemDepth = 12,
        extWgtP2MemDepth = 13,

        intWgtP1MemDepth = 380,
        intWgtDMemDepth = 20,
        intWgtP2MemDepth = 10,

        accMemDepth = 2048,
        outMemDepth = 1330,
        instQueueEntries = 512
      )
    ) ++ new WithAccelConfig)


class TensorBricksPynqConfigs(CONV: String = "PW", Hx: Int = 3, Wx: Int = 3, Cx: Int = 2, Cb: Int = 2, Fx: Int = 1, Px: Int = 1, K: Int = 3,
                              numPtrs: Int, numVals: Int, numRets: Int, numEvents: Int, numCtrls: Int)
  extends Config(new PynqConfig(CONV = CONV, Hx = Hx, Wx = Wx, Cx = Cx, Cb = Cb, Fx = Fx, Px = Px, K = K,
    num_ptrs = numPtrs, num_vals = numVals, num_event = numEvents, num_ctrl = numCtrls) ++
    new CoreConfig ++ new WithAccelConfig)

class DefaultDe10Config extends Config(new De10Config ++ new CoreConfig ++ new WithAccelConfig)

class DefaultPynqConfig extends Config(new PynqConfig ++ new CoreConfig ++ new WithAccelConfig)

object TensorBricksSimAccelMain extends App {

  var num_ptrs = 0
  var num_vals = 0
  var num_returns = 1
  var num_events = 1
  var num_ctrls = 1
  var Hx = 3
  var Wx = 3
  var Cx = 3
  var Cb = 3
  var Fx = 3
  var Px = 3
  var K = 3
  var CONV = "PW"

  args.sliding(2, 2).toList.collect {
    case Array("--num-ptrs", argPtrs: String) => num_ptrs = argPtrs.toInt
    case Array("--num-vals", argVals: String) => num_vals = argVals.toInt
    case Array("--num-rets", argRets: String) => num_returns = argRets.toInt
    case Array("--num-events", argEvent: String) => num_events = argEvent.toInt
    case Array("--num-ctrls", argCtrl: String) => num_ctrls = argCtrl.toInt

    case Array("--Hx", argCtrl: String) => Hx = argCtrl.toInt
    case Array("--Wx", argCtrl: String) => Wx = argCtrl.toInt
    case Array("--Cx", argCtrl: String) => Cx = argCtrl.toInt
    case Array("--Cb", argCtrl: String) => Cb = argCtrl.toInt
    case Array("--Fx", argCtrl: String) => Fx = argCtrl.toInt
    case Array("--Px", argCtrl: String) => Px = argCtrl.toInt
    case Array("--K", argCtrl: String) => K = argCtrl.toInt
    case Array("--CONV", argCtrl: String) => CONV = argCtrl
  }

  implicit val p: Parameters = new TensorBricksDe10Configs(CONV = CONV, Hx = Hx, Wx = Wx, Cx = Cx, Cb = Cb, Fx = Fx, Px = Px, K = K,
    numPtrs = num_ptrs, numVals = num_vals, numRets = num_returns, numEvents = num_events, numCtrls = num_ctrls)

  chisel3.Driver.execute(args.take(4), () => new TensorBricksSimAccel(CONV = CONV, Hx = Hx, Wx = Wx, Cx = Cx, Cb = Cb, Fx = Fx, Px = Px, K = K,
    numPtrs = num_ptrs, numVals = num_vals, numRets = num_returns, numEvents = num_events, numCtrls = num_ctrls))
}

object TensorBricksF1AccelMain extends App {

  var num_ptrs = 0
  var num_vals = 0
  var num_returns = 1
  var num_events = 4
  var num_ctrls = 1
  var Hx = 3
  var Wx = 3
  var Cx = 3
  var Cb = 3
  var Fx = 3
  var Px = 3
  var K = 3
  var CONV = "PW"

  args.sliding(2, 2).toList.collect {
    case Array("--num-ptrs", argPtrs: String) => num_ptrs = argPtrs.toInt
    case Array("--num-vals", argVals: String) => num_vals = argVals.toInt
    case Array("--num-rets", argRets: String) => num_returns = argRets.toInt
    case Array("--num-events", argEvent: String) => num_events = argEvent.toInt
    case Array("--num-ctrls", argCtrl: String) => num_ctrls = argCtrl.toInt

    case Array("--Hx", argCtrl: String) => Hx = argCtrl.toInt
    case Array("--Wx", argCtrl: String) => Wx = argCtrl.toInt
    case Array("--Cx", argCtrl: String) => Cx = argCtrl.toInt
    case Array("--Cb", argCtrl: String) => Cb = argCtrl.toInt
    case Array("--Fx", argCtrl: String) => Fx = argCtrl.toInt
    case Array("--Px", argCtrl: String) => Px = argCtrl.toInt
    case Array("--K", argCtrl: String) => K = argCtrl.toInt
    case Array("--CONV", argCtrl: String) => CONV = argCtrl.toString
  }

  implicit val p: Parameters = new TensorBricksF1Configs(CONV = CONV, Hx = Hx, Wx = Wx, Cx = Cx, Cb = Cb, Fx = Fx, Px = Px, K = K,
    numPtrs = num_ptrs, numVals = num_vals, numRets = num_returns, numEvents = num_events, numCtrls = num_ctrls)

  chisel3.Driver.execute(args.take(4), () => new F1Shell(CONV = CONV, Hx = Hx, Wx = Wx, Cx = Cx, Cb = Cb, Fx = Fx, Px = Px, K = K,
    numPtrs = num_ptrs, numVals = num_vals, numRets = num_returns, numEvents = num_events, numCtrls = num_ctrls))
}


object TestXilinxShellMain extends App {
  implicit val p: Parameters = new DefaultPynqConfig
  chisel3.Driver.execute(args, () => new XilinxShell())
}

object TestVTAShell2Main extends App {
  implicit val p: Parameters = new DefaultDe10Config
  chisel3.Driver.execute(args, () => new NoneAccel())
}

object DNNAccelMain extends App {
  implicit val p: Parameters = new DefaultDe10Config
  chisel3.Driver.execute(args, () => new DNNAccel())
}

