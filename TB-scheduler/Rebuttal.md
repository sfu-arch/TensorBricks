We thank the reviewers for their comments.

### Important Clarification [Rev D]. - “using inter-layer parallelism...overlapping the low-rank layers with other layers.”
There appears to be a mix-up; Tangram and Tensorbrick target orthogonal (but complementary) pieces in a DNN.

- Tensorbrick targets sequences of low-rank convs (depthwise and pointwise) that make up a composite block (e.g., inverse,inception).

  * Tangram pipelines layers with single CONV; they did not study composite blocks.
    Tensorbricks also pipelines CONV, but gains lower (e.g.,Figure 12:Resnet-50)

- Tensorbricks targets inference latency with batch size = 1.

  * Tangram targets throughput of large batches.

- Tensorbricks fuses outer-dataflow and maps onto a single partitioned engine (~784 MACs).

**Tensorbrick:Figure 9 can replace a single Eyeriss engine within Tangram. Tangram can leverage tensrobrick for fine-grain pipelining.**

### Common Q1 [A,D,E] . “Move the needle against state-of-the-art”

Tangram [8] and Fused [4]  fixed outer dataflow and blocking factor. `Tangram: (K=1. KCXY-CKXY)`. Fused-layer: `X=1,Y=1; XYKC-XYKC`.(details in Section 3.2).

In Figure 10-12 we swept 20,000 different  blocking factors  for each of these dataflows and picked best.

Comparing against the designs in [8] and [4]  directly.
|         | vs Tangram dataflow K=1.CKXY | vs Fusedlayer dataflow X=1,Y=1;XYKC |
| ------- | ---------------------------- | ----------------------------------- |
| Latency | 13.6x lower                  | 13.9x lower                         |
| Energy  | 126x lower                   | 0.90% lower                         |
| SRAM MB | 16.4x smaller                | 9x times smaller                    |

Resnet-50 (only spatial CONVs); Tangram dataflow performed 11% better.

### Common Q2 [A,B,D]. Evaluation Methodology?

Appendix includes details and microarchitecture constants.

For design space exploration we used an analytical model validated against hardware prototype on AWS F1 FPGA (see repo for Chisel RTL of tensorbrick).

For more details please refer repo figure here:

---

## Reviewer A

**Q1. Evaluation Methodology**

See common-Q2

**Q2. Definition and scope of Low Rank Convolution**

The workhorse blocks  (e.g., Inverse , Inception) in current DNNs are composite blocks, which internally include multiple depthwise and pointwise convs (each lower in rank). We refer to these convs as LR-CONVs.

**Q3. Importance of LR-CONVs**

|              | Composite blocks with LR-CONV layers      | Non-LR CONV                    |
| ------------ | ----------------------------------------- | ------------------------------ |
| Mobilenet-v2 | 22 layers.INV0--INV22 (3 LR-CONV per INV) | 3 layers. 1 CONV. 1 Pool. 1 FC |
| Xception     | 35 layers. DS0--DS35 (2 LR-CONVs per DS)  | 3 layers. 2 CONV. 1 FC         |

Imagenet leaderboard.
https://anonymous.4open.science/repository/12b2ace5-e3d3-47ea-b29d-46bc333c713e/tensorbricks_HW/Imagenet-leader.md

Top 10 networks spend more than 90%+ time in LR-CONVs Red markers indicate networks dominated by LR-CONVs.

**Q4. State of the art?**

See common-q1 above.

---

## Reviewer C

**Q1. Comparison to timeloop**

We will include timeloop; we cited “Understanding…”[MICRO’19] since it came later.

Timeloop does not model cross-layer dataflows (see https://accelergy.mit.edu/timeloop.pdf. Page 10:Section IX:``Future work includes….”.

`Figure 12: CXYK, XYCK,XYKC,KCXY` represent the dataflows timeloop will explore.

Tensorbrick dataflows, XYCK-PDP and XYCK-DP reduces latency by 20% and DRAM energy by 26%, with up to 50x lower SRAM.

**Q2. Design space exploration? **

We are the first to systematically expore:
- How to fuse outer-dataflows of adjacent layers and pipeline?
- How does the blocking factors of outer-dataflow affect pipeline initiation delays, stage latency and inter-layer buffering.
- How to allocate mac and SRAM resources non-uniformly across layers.

Further, we have validated the designs against a hardware prototype created on AWS F1 FPGA (Please see common-q2.)

--

## Reviewer E

**Q1. Area overhead is missing?**

We synthesize all designs and use FPGA for functional and hardware verification (see here for synthesis results). 

https://anonymous.4open.science/repository/12b2ace5-e3d3-47ea-b29d-46bc333c713e/tensorbricks_HW/Area.md


**Q2. Results for sweeping on chip memory and PEs?**
Fig 13: talks about non uniform mac allocation, for each layer.

Appendix: Figure 1, shows the on-chip memory size comparison after sweeping the design space.
We reduce SRAM needed by 3.5---50x

Table 1: On-chip SRAM requirement is controlled by Xb,Yb,Cb and Kb loop blocking of outer dataflow.

**Q4. Comparison against state of the art, baseline?**

See common-q1.

**Q5.  PE organization ?**

 Section 5.3 last paragraph page 10 explains in detail. Figure 13 fixed the dataflow to XYCK-PDP and analyzes the impact of non-uniform mac allocation.
This can have a further 75% impact on latency.

Design space exploration includes both blocking factors, Xb,Yb,Cb,Kb for all pipelined layers as well as non-uniform mac allocation for each layer. 

The hierarchical PE organization  helps vend macs in a fine-grain manner across layers. 

In INV blocks (Figure 13). DP (depthwise-CONVs) require only 2 L-Bricks (~100 macs), while PT-1 requires 3x more macs.

--

## Reviewer D

**Q1. what is the proposed solution**

Please see important clarification.

**Q2. Validation with the hardware**

Please see common-q2.

**Q3. Fit all parameters on chip**

We DO NOT assume all parameters on chip. The number of parameters on chip depends on the C_b and K_b parameters. See Fig 6 and 7.

**Q4. Increase bandwidth requirements? Intuition for outer dataflow, load balancing**

Please see animation here: https://anonymous.4open.science/repository/12b2ace5-e3d3-47ea-b29d-46bc333c713e/Tangram_vs_TB.md

Section 3.2 and Table:Figure 7 analyze Tangram’s dataflow: |K|C|XY in detail.

Tensorbrick advantages (X|Y|C|K)
- Can pipeline upto 3 stages high data reuse (beyond 2nd stage Tangram K|C|XY refetches from DRAM)
- Can explore blocking for X,Y,CK independently enabling low SRAM usae and incremental fetching from DRAM  (Tangram retains entire XY on-chip)
- Finer-grain partitioning of macs (allocates at brick 7x7 granularity) can better balance utilizatio   
  (Tangram allocates entire 256 mac eyeriss engines)

**Q6. Evaluation Methodology**

See common-q2

## Reviewer B

**Q1. Incremental to [4]**

See common q-1. 

We are the first to provide a tool to explore outer-dataflow fusion in detail.

Fusedlayer [4] employs XYKC-XYKC (X=1,Y=1) and vector hardware.
Fuselayer sacrifices parallelism, does not exploit filter  reuse and trade-offs storage for computation overhead.
 It serializes the loops in the XY  dimension.  As a result the features 	are not reused in XY dimension. 

Tensorbricks blocks  X,Y,C,K and uses 2D tensor hardware which can expl reuse in multiple dimensions (see hardware RTL here: https://anonymous.4open.science/repository/12b2ace5-e3d3-47ea-b29d-46bc333c713e/tensorbricks_HW/). 
For instance, our most optimal design point is
Xb=15,Yb=15,Cb=56,Fb=76; 8 L-Bricks(each 7x7 Macs)
Xb=Yb=13,Cb=56. 2 L-Bricks  
Xb=Yb=13,Cb=70,Fb=56.  6 L-Bricks 

**Q2 How does it compare to polyhedral?**

Polyhedral is completely orthogonal. It targets a single loop nest.
We pipeline across multiple loop nests, where each loop nest represents a low-rank conv.
---------------------

