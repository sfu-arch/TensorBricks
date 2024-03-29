package dnn.memory


import chisel3._
import chisel3.util._
import config._
import chipsalliance.rocketchip.config._
import shell._


/** Core parameters */
case class CoreParams(
    batch: Int = 1,
    blockOut: Int = 16,
    blockIn: Int = 8, //16,
    inpBits: Int = 32, //8,
    wgtBits: Int = 32, //8,

    kernelSize: Int = 9,
    PW1kernelSize: Int = 20,
    PW2kernelSize: Int = 20,

    uopBits: Int = 32,
    accBits: Int = 32,
    outBits: Int = 8,
    uopMemDepth: Int = 512,
    inpMemDepth: Int = 2660, //512

    wgtMemDepth: Int = 512,

    extWgtP1MemDepth: Int = 475,
    extWgtDMemDepth: Int = 12,
    extWgtP2MemDepth: Int = 13,

    intWgtP1MemDepth: Int = 380,
    intWgtDMemDepth: Int = 20,
    intWgtP2MemDepth: Int = 10,

    accMemDepth: Int = 512,
    outMemDepth: Int = 1330,//512,
    instQueueEntries: Int = 32
) {
  require(uopBits % 8 == 0,
          s"\n\n[VTA] [CoreParams] uopBits must be byte aligned\n\n")
}

case object CoreKey extends Field[CoreParams]

/** Core.
  *
  * The core defines the current VTA architecture by connecting memory and
  * compute modules together such as load/store and compute. Most of the
  * connections in the core are bulk (<>), and we should try to keep it this
  * way, because it is easier to understand what is going on.
  *
  * Also, the core must be instantiated by a shell using the
  * VTA Control Register (VCR) and the VTA Memory Engine (VME) interfaces.
  * More info about these interfaces and modules can be found in the shell
  * directory.
  */
class CoreConfig(inParams: CoreParams =
                 CoreParams(
                   batch = 1,
                   blockOut = 16,
                   blockIn = 8, //16,
                   inpBits = 32,//8,
                   wgtBits = 32,//8,
                   kernelSize = 9,
                   PW1kernelSize = 5,
                   PW2kernelSize = 5,
                   uopBits = 32,
                   accBits = 32,
                   outBits = 8,
                   uopMemDepth = 2048,
                   inpMemDepth = 2048, //2660
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
                )
    extends Config((site, here, up) => {
      case CoreKey => inParams

    })