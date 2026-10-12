/* -----------------------------------------------------------------------------
AlfProfiler.h - time / energy accounting of the ALF stages inside VVenC and the
shared GPU backend instance.

Enabled with --AlfProf=1. Stage scopes are placed in EncSlice around the ALF
tasks; with several encoder threads the per-stage numbers are *sums over
threads* (CPU-time-like), the encoder wall time is reported separately.

"Whole ALF" = everything the encoder does for ALF + CCALF, from the start of the
statistics to the end of the reconstruction: classification (CPU or GPU), covariance
statistics, Wiener filter derivation, CTU on/off and filter-set decisions, APS,
luma/chroma filtering, CCALF. Its time is the sum of the stage scopes and, as wall
clock, the time during which at least one ALF scope is active; its energy is RAPL
inside the scopes (CPU) and NVML power integrated only while ALF is active (GPU).

Energy sources:
  CPU : Linux powercap / RAPL  (/sys/class/powercap/intel-rapl:N/energy_uj, package domains)
  GPU : NVML total energy counter (Volta+) and a power-sampling thread (fallback / avg/peak)
------------------------------------------------------------------------------- */
#pragma once

#include "CommonLib/AlfGpu.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace vvenc
{

enum AlfProfStage
{
  ALFP_STATS = 0,        // ALF_GET_STATISTICS  (classification + covariance statistics)
  ALFP_STATS_CLASS_CPU,  //   luma 4x4 classification on the CPU (VVenC SIMD), part of ALFP_STATS
  ALFP_STATS_CLASS_GPU,  //   --AlfGpu 1/3: gradients + classes + transpose of the whole picture on the GPU (call wall time), part of ALFP_STATS
  ALFP_STATS_COV_CPU,    //   covariance statistics on the CPU (VVenC getPreBlkStats, SIMD), part of ALFP_STATS
  ALFP_STATS_COV_GPU,    //   --AlfGpu 1/3: covariance statistics of the whole picture on the GPU (call wall time + unpack), part of ALFP_STATS
  ALFP_DERIVE,           // ALF_DERIVE_FILTER   (filter derivation, CTU decisions, APS)
  ALFP_RECON,            // ALF_RECONSTRUCT     (whole stage: luma + chroma)
  ALFP_RECON_LUMA_CPU,   //   luma 7x7 filtering on the CPU (VVenC SIMD)
  ALFP_RECON_LUMA_GPU,   //   luma band call to the GPU backend (wall clock)
  ALFP_CCALF_STATS,
  ALFP_CCALF_DERIVE,
  ALFP_CCALF_RECON,
  ALFP_NUM
};

class RaplReader
{
public:
  RaplReader();
  ~RaplReader();
  bool     ok() const { return !m_fds.empty(); }
  uint64_t readUJ();                // sum of all package domains, wrap-around corrected, monotonic
private:
  std::vector<int>      m_fds;
  std::vector<uint64_t> m_max, m_last, m_acc;
  std::mutex            m_mutex;
};

class AlfProfiler
{
public:
  static AlfProfiler& get();

  bool enabled() const { return m_enabled; }
  void start( bool enable, int gpuMode );       // EncLib::initEncoderLib
  void stop();                                  // EncLib::uninitEncoderLib, prints the report

  // shared backend (created on first use with the fixed filter tables)
  alfgpu::Backend* backend( const int16_t* fixedCoef, const int16_t* fixedClip, int maxPicWidth, int maxBandH, int numWorkers );

  void addStage   ( AlfProfStage s, double ms, uint64_t raplUJ );
  void addGpu     ( const alfgpu::Timing& t );
  void addAlfPicture() { m_alfPictures++; }   // pictures not skipped by isSkipAlfForFrame()
  void addEncPicture() { m_encPictures++; }   // every picture that reaches the ALF derivation
  // whole-ALF wall-clock tracking (top-level stage scopes)
  void alfEnter();
  void alfLeave();
  void addVerify  ( uint64_t samples, uint64_t sampleErr, uint64_t blocks, uint64_t blockErr );
  void addVerifyCov( uint64_t values, uint64_t errors );
  uint64_t rapl() { return m_rapl.ok() ? m_rapl.readUJ() : 0; }

private:
  AlfProfiler() {}
  ~AlfProfiler();
  void sampler();
  void openGpuWindow();
  void report( FILE* f );

  bool m_enabled = false;
  int  m_gpuMode = 0;
  std::chrono::steady_clock::time_point m_t0, m_t1, m_t1Real;

  std::mutex        m_mutex;
  double            m_stageMs[ALFP_NUM]    = { 0 };
  uint64_t          m_stageCalls[ALFP_NUM] = { 0 };
  uint64_t          m_stageUJ[ALFP_NUM]    = { 0 };
  alfgpu::Timing    m_gpu;
  uint64_t          m_gpuCalls = 0;
  std::atomic<uint64_t> m_alfPictures{ 0 }, m_encPictures{ 0 };
  double alfActiveS();                         // seconds with >= 1 ALF scope active, including the running interval
  std::mutex        m_activeMutex;
  int               m_activeCount = 0;
  std::chrono::steady_clock::time_point m_activeT0;
  double            m_activeS = 0;
  double            m_gpuAlfJ = 0;             // GPU energy integrated while ALF is active
  uint64_t          m_verSamples = 0, m_verSampleErr = 0, m_verBlocks = 0, m_verBlockErr = 0, m_verCov = 0, m_verCovErr = 0;

  alfgpu::Backend*  m_backend = nullptr;
  std::mutex        m_backendMutex;

  RaplReader        m_rapl;
  uint64_t          m_cpuUJ0 = 0, m_cpuUJ1 = 0;
  double            m_cpuIdleW = -1;

  // GPU power sampling
  std::thread             m_samplerThread;
  std::atomic<bool>       m_samplerRun{ false };
  double                  m_gpuSampledJ = 0, m_gpuPeakW = 0, m_gpuIdleW = -1;
  unsigned long long      m_gpuE0 = 0, m_gpuE1 = 0;
  std::chrono::steady_clock::time_point m_gpuT0;
  bool                    m_gpuWindowOpen = false;
  double                  m_excludedS = 0;
  uint64_t                m_excludedUJ = 0;   // idle sampling done inside the encoding window
  bool                    m_gpuEnergyOk = false;
};

// RAII scope: wall time (+ RAPL delta for the stage-level scopes) of an ALF stage.
// The RAPL reads happen outside the timed interval; fine-grained (per CTU) scopes skip RAPL.
class AlfProfScope
{
public:
  AlfProfScope( AlfProfStage s ) : m_s( s ), m_on( AlfProfiler::get().enabled() ),
                                   m_energy( s != ALFP_RECON_LUMA_CPU && s != ALFP_RECON_LUMA_GPU && s != ALFP_STATS_CLASS_CPU && s != ALFP_STATS_CLASS_GPU
                                             && s != ALFP_STATS_COV_CPU && s != ALFP_STATS_COV_GPU )
  {
    // stage-level scopes (m_energy) are the top-level ALF scopes: RAPL + whole-ALF wall clock
    if( m_on ) { if( m_energy ) { m_uj = AlfProfiler::get().rapl(); AlfProfiler::get().alfEnter(); } m_t = std::chrono::steady_clock::now(); }
  }
  ~AlfProfScope()
  {
    if( !m_on ) return;
    const double ms = std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - m_t ).count();
    if( m_energy ) AlfProfiler::get().alfLeave();
    uint64_t duj = 0;
    if( m_energy ) { const uint64_t uj = AlfProfiler::get().rapl(); duj = uj >= m_uj ? uj - m_uj : 0; }
    AlfProfiler::get().addStage( m_s, ms, duj );
  }
private:
  AlfProfStage m_s;
  bool m_on, m_energy;
  uint64_t m_uj = 0;
  std::chrono::steady_clock::time_point m_t;
};

} // namespace vvenc
