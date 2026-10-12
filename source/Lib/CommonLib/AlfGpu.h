/* -----------------------------------------------------------------------------
AlfGpu.h  -  backend interface for offloading the ALF luma reconstruction

The encoder hands over one "band" (one CTU row, or several) at a time. The band
contains the pre-ALF luma samples (m_tempBuf, already border-extended by VVenC),
the list of CTUs to filter with their filter-set index, and the APS filter sets of
the current picture. The backend writes the filtered samples straight into the
destination (picture reconstruction) buffer for those CTUs only.

Two backends implement it:
  * "cuda"      - real GPU kernels (AlfGpuCuda.cu, built with -DVVENC_ENABLE_CUDA_ALF=ON)
  * "host-emu"  - the same data flow and the same core functions run on the CPU;
                  used to validate bit-exactness without a GPU.
Like AlfGpuCore.h, this header has no VVenC dependencies.
------------------------------------------------------------------------------- */
#pragma once

#include "AlfGpuTiles.h"
#include "AlfGpuCov.h"

#include <atomic>

namespace alfgpu
{

enum Mode : int
{
  MODE_OFF        = 0,   // original VVenC CPU path
  MODE_FULL       = 1,   // GPU gradients/classification/transpose (at the start of the ALF, also used for the
                         // encoder statistics) + GPU filtering with the geometric transformation
  MODE_FILTER     = 2,   // GPU filtering, classification taken from VVenC (m_classifier)
  MODE_VERIFY     = 3,   // MODE_FULL + CPU reference, compare sample by sample (keeps CPU result)
};

struct BandParams
{
  // input: pre-ALF luma (m_tempBuf). src points to picture sample (0, bandY0);
  // samples [-kHalo, picWidth + kHalo) x [bandY0 - kHalo, bandY0 + bandH + kHalo) must be readable
  const Pel*  src;
  ptrdiff_t   srcStride;

  int picWidth;          // luma width (pcv.lumaWidth)
  int bandY0;            // first luma row of the band (multiple of ctuSize)
  int bandH;             // rows in the band
  int ctuSize;
  int numCtuCols;        // CTU columns in the picture
  int bitDepth;
  int vbH, vbPos;        // ALF virtual boundary parameters

  // per CTU of the band (raster inside the band, numCtuCols * ceil(bandH/ctuSize)):
  // filter set index (0..15 fixed, 16.. APS) or -1 = not filtered by the backend
  const int16_t* ctuSetIdx;

  // APS sets referenced by setIdx >= 16: numAps * kSetStride coefficients and clip values
  const int16_t* apsCoef;
  const int16_t* apsClip;
  int numAps;

  // MODE_FILTER: classification from VVenC, one byte per 4x4 block, (classIdx | transposeIdx << 5),
  // row-major with stride picWidth/4, bandH/4 rows. Unused otherwise.
  const uint8_t* cpuClass;

  // output: dst points to picture sample (0, bandY0) of the reconstruction buffer
  Pel*      dst;
  ptrdiff_t dstStride;

  // optional: if not null, receives the backend's classification (same packing as cpuClass)
  uint8_t*  outClass;
};

// covariance statistics of one colour component for a set of statistics units (ASUs or CTUs)
struct CovJob
{
  int comp;                    // 0 = luma: uses the picture and classes resident from classifyFrame(); 1, 2 = chroma
  const Pel* rec;              // chroma: pre-ALF plane (m_tempBuf) at sample (0,0), kHalo samples readable around it
  ptrdiff_t  recStride;
  const Pel* org;              // original plane at (0,0)
  ptrdiff_t  orgStride;
  int width, height;           // plane size
  CovKParams k;                // shape, bins, clipping values, virtual boundary (recPitch/orgPitch/clsStride set by the backend)
  int numUnits;
  const int*     unitCtuStart; // numUnits + 1 entries into ctuRect
  const int16_t* ctuRect;      // x0, y0, w, h of each CTU (plane coordinates), CPU order inside each unit
  const uint32_t* elem;        // k.numElem element descriptors (covElements())
  float*   out;                // numUnits * numClasses * numElem accumulated values
  uint8_t* used;               // numUnits * numClasses: class received at least one block (AlfCovariance::all0 == false)
};

struct Timing
{
  double hostPackMs   = 0;   // pageable -> staging copies (input)
  double h2dMs        = 0;   // host -> device transfers
  double classifyMs   = 0;   // classification kernel
  double filterMs     = 0;   // filtering kernel
  double covMs        = 0;   // covariance statistics kernels
  double d2hMs        = 0;   // device -> host transfers
  double hostUnpackMs = 0;   // staging -> picture buffer copies (output)
  double gpuSpanMs    = 0;   // GPU timeline from the first H2D to the last D2H (stages overlap across streams)
  double totalMs      = 0;   // wall clock of the whole call
  int    chunks       = 0;   // pipeline chunks used
  uint64_t bytesH2D   = 0;
  uint64_t bytesD2H   = 0;
};

class Backend
{
  std::atomic<int> m_nextSlot{ 0 };
public:
  virtual ~Backend() {}
  virtual const char* name() const = 0;
  // fixedCoef/fixedClip: kNumFixedSets * kSetStride each (m_fixedFilterSetCoeffDec / m_clipDefault)
  virtual bool init( const int16_t* fixedCoef, const int16_t* fixedClip, int maxPicWidth, int maxBandH, int numWorkers ) = 0;
  // thread-safe for distinct worker indices
  virtual void filterLumaBand( const BandParams& p, int mode, int worker, Timing& t ) = 0;

  // ---- frame-resident path: one slot per encoder picture context (EncAdaptiveLoopFilter instance)
  // Stage 1, at the start of the ALF (ALF_GET_STATISTICS): uploads the whole picture (p.src, bandY0 = 0,
  //   bandH = picture height), computes gradients, class and transpose index of every 4x4 block of the CTUs
  //   with p.ctuSetIdx >= 0 and downloads them to p.outClass. Picture and classes stay on the device.
  virtual void classifyFrame( const BandParams& p, int slot, Timing& t ) = 0;
  // Stage 2, after the filter decisions (ALF_DERIVE_FILTER): filters the CTUs with p.ctuSetIdx >= 0 (filter set
  //   index) with the resident picture and classes; p.src / p.cpuClass are not used. Output to p.dst.
  virtual void filterFrameResident( const BandParams& p, int slot, Timing& t ) = 0;
  // Statistics (ALF_GET_STATISTICS), after classifyFrame() on the same slot: covariance statistics of the jobs
  //   (luma reuses the resident picture and classes; chroma and the original planes are uploaded).
  virtual void covarianceFrame( const CovJob* jobs, int numJobs, int slot, Timing& t ) = 0;
  int  acquireSlot() { return m_nextSlot++; }
  // GPU energy counter (NVML, mJ since driver load) and instantaneous power (mW); false if unavailable
  virtual bool readEnergy_mJ( unsigned long long& )  { return false; }
  virtual bool readPower_mW ( unsigned& )            { return false; }
};

// returns an initialised CUDA backend when compiled with VVENC_CUDA_ALF and a device is present
// (unless env VVENC_ALF_GPU_EMU=1), otherwise the initialised host emulation backend
Backend* createBackend( const int16_t* fixedCoef, const int16_t* fixedClip, int maxPicWidth, int maxBandH, int numWorkers );

// host reference used by the emulation backend and by the benchmark (exposed for tests)
void hostClassifyBand( const Pel* in, ptrdiff_t inStride, const BandParams& p, uint8_t* cls, int clsStride );
void hostFilterBand  ( const Pel* in, ptrdiff_t inStride, const BandParams& p, const int16_t* coefTab, const int16_t* clipTab,
                       const uint8_t* cls, int clsStride, Pel* out, ptrdiff_t outStride );

} // namespace alfgpu
