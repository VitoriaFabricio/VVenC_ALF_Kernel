/* -----------------------------------------------------------------------------
AlfProfiler.cpp - see AlfProfiler.h
------------------------------------------------------------------------------- */

#include "AlfProfiler.h"

#include <cinttypes>
#include <cstdlib>
#include <cstring>

#if defined( __linux__ )
#  include <dirent.h>
#  include <fcntl.h>
#  include <unistd.h>
#endif

namespace vvenc
{

// ====================================================================================================================
// RAPL (Linux powercap). Needs read permission on energy_uj (root, or: sudo chmod o+r /sys/class/powercap/intel-rapl:*/energy_uj)
// ====================================================================================================================

RaplReader::RaplReader()
{
#if defined( __linux__ )
  for( int pkg = 0; pkg < 16; pkg++ )
  {
    const std::string base = "/sys/class/powercap/intel-rapl:" + std::to_string( pkg );
    const int fd = open( ( base + "/energy_uj" ).c_str(), O_RDONLY );
    if( fd < 0 ) continue;
    char buf[64] = { 0 };
    if( pread( fd, buf, sizeof( buf ) - 1, 0 ) <= 0 ) { close( fd ); continue; }
    uint64_t maxr = 0;
    if( FILE* f = fopen( ( base + "/max_energy_range_uj" ).c_str(), "r" ) )
    {
      if( fscanf( f, "%" SCNu64, &maxr ) != 1 ) maxr = 0;
      fclose( f );
    }
    m_fds.push_back( fd );
    m_max.push_back( maxr );
    m_last.push_back( strtoull( buf, nullptr, 10 ) );
    m_acc.push_back( 0 );
  }
#endif
}

RaplReader::~RaplReader()
{
#if defined( __linux__ )
  for( int fd : m_fds ) close( fd );
#endif
}

uint64_t RaplReader::readUJ()
{
  uint64_t sum = 0;
#if defined( __linux__ )
  std::lock_guard<std::mutex> lock( m_mutex );
  for( size_t i = 0; i < m_fds.size(); i++ )
  {
    char buf[64] = { 0 };
    if( pread( m_fds[i], buf, sizeof( buf ) - 1, 0 ) > 0 )
    {
      const uint64_t v = strtoull( buf, nullptr, 10 );
      const uint64_t d = v >= m_last[i] ? v - m_last[i] : v + m_max[i] - m_last[i];   // counter wrap
      m_acc[i] += d;
      m_last[i] = v;
    }
    sum += m_acc[i];
  }
#endif
  return sum;
}

// ====================================================================================================================

AlfProfiler& AlfProfiler::get()
{
  static AlfProfiler p;
  return p;
}

AlfProfiler::~AlfProfiler()
{
  if( m_samplerRun ) { m_samplerRun = false; if( m_samplerThread.joinable() ) m_samplerThread.join(); }
  // the backend is intentionally not deleted here: CUDA may already be torn down at static destruction time
}

alfgpu::Backend* AlfProfiler::backend( const int16_t* fixedCoef, const int16_t* fixedClip, int maxPicWidth, int maxBandH, int numWorkers )
{
  std::lock_guard<std::mutex> lock( m_backendMutex );
  if( !m_backend )
  {
    m_backend = alfgpu::createBackend( fixedCoef, fixedClip, maxPicWidth, maxBandH, numWorkers );
    fprintf( stderr, "[AlfGpu] backend: %s\n", m_backend->name() );
    if( m_enabled ) openGpuWindow();
  }
  return m_backend;
}

void AlfProfiler::openGpuWindow()
{
  if( !m_backend || m_samplerRun || m_gpuWindowOpen ) return;
  m_gpuWindowOpen = true;
  const auto tIdle0 = std::chrono::steady_clock::now();
  const uint64_t ujIdle0 = rapl();
  unsigned p;
  if( m_backend->readPower_mW( p ) )
  {
    // GPU idle baseline (power after context creation, before any ALF work)
    int idleMs = 500;
    if( const char* s = getenv( "VVENC_ALF_PROF_IDLE_MS" ) ) idleMs = std::max( 0, atoi( s ) );
    double acc = 0; int n = 0;
    for( int t = 0; t < idleMs; t += 10 )
    {
      std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
      if( m_backend->readPower_mW( p ) ) { acc += p * 1e-3; n++; }
    }
    m_gpuIdleW = n > 0 ? acc / n : -1;
  }
  // idle sampling must not count as encoding time
  m_excludedS  += std::chrono::duration<double>( std::chrono::steady_clock::now() - tIdle0 ).count();
  m_excludedUJ += rapl() - ujIdle0;

  unsigned long long e;
  m_gpuEnergyOk = m_backend->readEnergy_mJ( e );
  m_gpuE0 = m_gpuEnergyOk ? e : 0;
  m_gpuT0 = std::chrono::steady_clock::now();
  if( m_backend->readPower_mW( p ) )
  {
    m_samplerRun    = true;
    m_samplerThread = std::thread( &AlfProfiler::sampler, this );
  }
}

void AlfProfiler::sampler()
{
  using clk = std::chrono::steady_clock;
  auto last = clk::now();
  unsigned p0 = 0;
  m_backend->readPower_mW( p0 );
  double a0 = alfActiveS();
  while( m_samplerRun )
  {
    std::this_thread::sleep_for( std::chrono::milliseconds( 2 ) );
    unsigned p1 = 0;
    if( !m_backend->readPower_mW( p1 ) ) continue;
    const auto now = clk::now();
    const double a1 = alfActiveS();
    const double dt = std::chrono::duration<double>( now - last ).count();
    const double pW = 0.5 * ( p0 + p1 ) * 1e-3;                 // trapezoid
    m_gpuSampledJ += pW * dt;
    m_gpuAlfJ     += pW * std::min( dt, a1 - a0 );             // only the part of the interval with ALF active
    m_gpuPeakW     = std::max( m_gpuPeakW, p1 * 1e-3 );
    p0 = p1; last = now; a0 = a1;
  }
}

void AlfProfiler::start( bool enable, int gpuMode )
{
  m_enabled = enable;
  m_gpuMode = gpuMode;
  if( !enable ) return;

  // idle baseline of the CPU package (and GPU, if the backend already exists) before encoding
  int idleMs = 500;
  if( const char* s = getenv( "VVENC_ALF_PROF_IDLE_MS" ) ) idleMs = std::max( 0, atoi( s ) );
  if( idleMs > 0 && m_rapl.ok() )
  {
    const uint64_t e0 = m_rapl.readUJ();
    const auto t0 = std::chrono::steady_clock::now();
    std::this_thread::sleep_for( std::chrono::milliseconds( idleMs ) );
    const uint64_t e1 = m_rapl.readUJ();
    const double s = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
    m_cpuIdleW = ( e1 - e0 ) * 1e-6 / s;
  }
  if( !m_rapl.ok() )
  {
    fprintf( stderr, "[AlfProf] RAPL not readable (no /sys/class/powercap/intel-rapl:N/energy_uj or no permission) - CPU energy disabled\n" );
  }
  m_cpuUJ0 = m_rapl.ok() ? m_rapl.readUJ() : 0;
  m_t0 = std::chrono::steady_clock::now();
  {
    std::lock_guard<std::mutex> lock( m_backendMutex );
    openGpuWindow();   // backend may already exist
  }
}

void AlfProfiler::alfEnter()
{
  std::lock_guard<std::mutex> lock( m_activeMutex );
  if( m_activeCount++ == 0 ) m_activeT0 = std::chrono::steady_clock::now();
}

void AlfProfiler::alfLeave()
{
  std::lock_guard<std::mutex> lock( m_activeMutex );
  if( --m_activeCount == 0 ) m_activeS += std::chrono::duration<double>( std::chrono::steady_clock::now() - m_activeT0 ).count();
}

double AlfProfiler::alfActiveS()
{
  std::lock_guard<std::mutex> lock( m_activeMutex );
  double s = m_activeS;
  if( m_activeCount > 0 ) s += std::chrono::duration<double>( std::chrono::steady_clock::now() - m_activeT0 ).count();
  return s;
}

void AlfProfiler::addStage( AlfProfStage s, double ms, uint64_t uj )
{
  std::lock_guard<std::mutex> lock( m_mutex );
  m_stageMs[s] += ms;
  m_stageCalls[s]++;
  m_stageUJ[s] += uj;
}

void AlfProfiler::addGpu( const alfgpu::Timing& t )
{
  std::lock_guard<std::mutex> lock( m_mutex );
  m_gpu.hostPackMs   += t.hostPackMs;
  m_gpu.h2dMs        += t.h2dMs;
  m_gpu.classifyMs   += t.classifyMs;
  m_gpu.filterMs     += t.filterMs;
  m_gpu.covMs        += t.covMs;
  m_gpu.d2hMs        += t.d2hMs;
  m_gpu.hostUnpackMs += t.hostUnpackMs;
  m_gpu.totalMs      += t.totalMs;
  m_gpu.gpuSpanMs    += t.gpuSpanMs;
  m_gpu.chunks       += t.chunks;
  m_gpu.bytesH2D     += t.bytesH2D;
  m_gpu.bytesD2H     += t.bytesD2H;
  m_gpuCalls++;
}

void AlfProfiler::addVerify( uint64_t samples, uint64_t sampleErr, uint64_t blocks, uint64_t blockErr )
{
  std::lock_guard<std::mutex> lock( m_mutex );
  m_verSamples += samples; m_verSampleErr += sampleErr; m_verBlocks += blocks; m_verBlockErr += blockErr;
}

void AlfProfiler::addVerifyCov( uint64_t values, uint64_t errors )
{
  std::lock_guard<std::mutex> lock( m_mutex );
  m_verCov += values; m_verCovErr += errors;
}

void AlfProfiler::stop()
{
  if( m_verSamples || m_verBlocks || m_verCov )
  {
    // verification is reported even without --AlfProf
    fprintf( stderr, "[AlfGpu] VERIFY: %" PRIu64 " samples, %" PRIu64 " mismatches | %" PRIu64 " 4x4 classes, %" PRIu64 " mismatches | %" PRIu64
                     " covariance values, %" PRIu64 " mismatches -> %s\n",
             m_verSamples, m_verSampleErr, m_verBlocks, m_verBlockErr, m_verCov, m_verCovErr,
             ( m_verSampleErr || m_verBlockErr || m_verCovErr ) ? "NOT BIT-EXACT" : "BIT-EXACT" );
  }
  if( !m_enabled ) return;
  m_t1Real = std::chrono::steady_clock::now();
  m_t1 = m_t1Real - std::chrono::duration_cast<std::chrono::steady_clock::duration>( std::chrono::duration<double>( m_excludedS ) );
  m_cpuUJ1 = m_rapl.ok() ? m_rapl.readUJ() - m_excludedUJ : 0;
  if( m_samplerRun ) { m_samplerRun = false; if( m_samplerThread.joinable() ) m_samplerThread.join(); }
  if( m_backend && m_gpuEnergyOk )
  {
    unsigned long long e;
    if( m_backend->readEnergy_mJ( e ) ) m_gpuE1 = e; else m_gpuEnergyOk = false;
  }
  report( stderr );
  if( const char* csv = getenv( "VVENC_ALF_PROF_CSV" ) )
  {
    FILE* f = fopen( csv, "a" );
    if( f )
    {
      fseek( f, 0, SEEK_END );
      if( ftell( f ) == 0 )
      {
        fprintf( f, "label,backend,mode,enc_s,stats_ms,derive_ms,recon_ms,luma_cpu_ms,luma_gpu_ms,ccalf_ms,"
                    "gpu_calls,pack_ms,h2d_ms,classify_ms,filter_ms,d2h_ms,unpack_ms,gpu_total_ms,bytes_h2d,bytes_d2h,"
                    "cpu_pkg_J,cpu_idle_W,alf_scope_cpu_J,gpu_J,gpu_idle_W,gpu_sampled_J,gpu_peak_W,gpu_span_ms,gpu_chunks,class_cpu_ms,alf_pictures,enc_pictures,class_gpu_ms,whole_alf_ms,whole_alf_wall_ms,whole_alf_cpu_J,whole_alf_gpu_J,cov_cpu_ms,cov_gpu_ms,cov_kernel_ms\n" );
      }
      const char* label = getenv( "VVENC_ALF_PROF_LABEL" );
      const double wholeMs = m_stageMs[ALFP_STATS] + m_stageMs[ALFP_DERIVE] + m_stageMs[ALFP_RECON]
                           + m_stageMs[ALFP_CCALF_STATS] + m_stageMs[ALFP_CCALF_DERIVE] + m_stageMs[ALFP_CCALF_RECON];
      const double wholeCpuJ = m_rapl.ok() ? ( m_stageUJ[ALFP_STATS] + m_stageUJ[ALFP_DERIVE] + m_stageUJ[ALFP_RECON]
                             + m_stageUJ[ALFP_CCALF_STATS] + m_stageUJ[ALFP_CCALF_DERIVE] + m_stageUJ[ALFP_CCALF_RECON] ) * 1e-6 : -1.0;
      const double encS = std::chrono::duration<double>( m_t1 - m_t0 ).count();
      fprintf( f, "%s,%s,%d,%.6f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%" PRIu64 ",%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%" PRIu64 ",%" PRIu64 ",%.4f,%.3f,%.4f,%.4f,%.3f,%.4f,%.2f,%.3f,%d,%.3f,%d,%d,%.3f,%.3f,%.3f,%.5f,%.5f,%.3f,%.3f,%.3f\n",
               label ? label : "", m_backend ? m_backend->name() : "cpu", m_gpuMode, encS,
               m_stageMs[ALFP_STATS], m_stageMs[ALFP_DERIVE], m_stageMs[ALFP_RECON], m_stageMs[ALFP_RECON_LUMA_CPU], m_stageMs[ALFP_RECON_LUMA_GPU],
               m_stageMs[ALFP_CCALF_STATS] + m_stageMs[ALFP_CCALF_DERIVE] + m_stageMs[ALFP_CCALF_RECON],
               m_gpuCalls, m_gpu.hostPackMs, m_gpu.h2dMs, m_gpu.classifyMs, m_gpu.filterMs, m_gpu.d2hMs, m_gpu.hostUnpackMs, m_gpu.totalMs,
               m_gpu.bytesH2D, m_gpu.bytesD2H,
               ( m_cpuUJ1 - m_cpuUJ0 ) * 1e-6, m_cpuIdleW,
               ( m_stageUJ[ALFP_STATS] + m_stageUJ[ALFP_DERIVE] + m_stageUJ[ALFP_RECON] ) * 1e-6,
               m_gpuEnergyOk ? ( m_gpuE1 - m_gpuE0 ) * 1e-3 : -1.0, m_gpuIdleW, m_gpuSampledJ, m_gpuPeakW,
               m_gpu.gpuSpanMs, ( int ) m_gpu.chunks, m_stageMs[ALFP_STATS_CLASS_CPU], ( int ) m_alfPictures,
               ( int ) m_encPictures, m_stageMs[ALFP_STATS_CLASS_GPU], wholeMs, alfActiveS() * 1e3, wholeCpuJ, m_backend ? m_gpuAlfJ : -1.0,
               m_stageMs[ALFP_STATS_COV_CPU], m_stageMs[ALFP_STATS_COV_GPU], m_gpu.covMs );
      fclose( f );
    }
  }
}

void AlfProfiler::report( FILE* f )
{
  const double encS  = std::chrono::duration<double>( m_t1 - m_t0 ).count();
  const double encMs = encS * 1e3;
  static const char* names[ALFP_NUM] = { "ALF stats (class+cov)", "  luma classif. CPU", "  luma classif. GPU call", "  covariance CPU", "  covariance GPU call", "ALF derive filter", "ALF reconstruct (Y+C)",
                                          "  luma filter CPU", "  luma filter GPU call", "CCALF stats", "CCALF derive", "CCALF reconstruct" };

  fprintf( f, "\n================ ALF profile ================\n" );
  fprintf( f, "backend: %s | AlfGpu mode: %d | encoder wall time: %.3f s\n", m_backend ? m_backend->name() : "cpu (VVenC)", m_gpuMode, encS );
  fprintf( f, "(stage times are summed over encoder threads; run with --Threads 0 or 1 for a clean single-thread profile)\n\n" );
  fprintf( f, "%-24s %8s %12s %8s %12s %12s\n", "stage", "calls", "time [ms]", "% enc", "avg [ms]", "RAPL [J]" );
  double alfMs = 0;
  for( int s = 0; s < ALFP_NUM; s++ )
  {
    if( !m_stageCalls[s] ) continue;
    const bool fine = s == ALFP_RECON_LUMA_CPU || s == ALFP_RECON_LUMA_GPU || s == ALFP_STATS_CLASS_CPU || s == ALFP_STATS_CLASS_GPU
                    || s == ALFP_STATS_COV_CPU || s == ALFP_STATS_COV_GPU;   // no RAPL inside per-CTU scopes
    fprintf( f, "%-24s %8" PRIu64 " %12.3f %7.2f%% %12.4f", names[s], m_stageCalls[s], m_stageMs[s],
             100.0 * m_stageMs[s] / encMs, m_stageMs[s] / m_stageCalls[s] );
    if( fine ) fprintf( f, " %12s\n", "-" ); else fprintf( f, " %12.4f\n", m_stageUJ[s] * 1e-6 );
    if( s == ALFP_STATS || s == ALFP_DERIVE || s == ALFP_RECON || s >= ALFP_CCALF_STATS ) alfMs += m_stageMs[s];
  }
  fprintf( f, "%-24s %8s %12.3f %7.2f%%\n", "ALF+CCALF total", "", alfMs, 100.0 * alfMs / encMs );
  // ------------------------------------------------------------------ whole ALF
  const double activeS = alfActiveS();
  const double cpuAlfJ = ( m_stageUJ[ALFP_STATS] + m_stageUJ[ALFP_DERIVE] + m_stageUJ[ALFP_RECON]
                         + m_stageUJ[ALFP_CCALF_STATS] + m_stageUJ[ALFP_CCALF_DERIVE] + m_stageUJ[ALFP_CCALF_RECON] ) * 1e-6;
  const bool   gpuE    = m_backend && m_samplerRun.load() == false && m_gpuSampledJ > 0;
  {
    const double np = ( double ) std::max<uint64_t>( 1, m_encPictures ), na = ( double ) std::max<uint64_t>( 1, m_alfPictures );
    fprintf( f, "\nWHOLE ALF  (start of the statistics -> end of the reconstruction: classification, covariance statistics,\n" );
    fprintf( f, "            Wiener derivation, CTU/filter-set decisions, APS, luma+chroma filtering, CCALF)\n" );
    fprintf( f, "                                              total        per picture   per picture with ALF\n" );
    fprintf( f, "  pictures                               %10" PRIu64 "                     %10" PRIu64 "\n", ( uint64_t ) m_encPictures, ( uint64_t ) m_alfPictures );
    fprintf( f, "  time, sum of ALF scopes [ms]           %10.3f   %12.4f   %12.4f   (%.2f%% of the encoder)\n", alfMs, alfMs / np, alfMs / na, 100.0 * alfMs / encMs );
    fprintf( f, "  time, wall clock with ALF active [ms]  %10.3f   %12.4f   %12.4f\n", activeS * 1e3, activeS * 1e3 / np, activeS * 1e3 / na );
    fprintf( f, "    of which luma classification [ms]    %10.3f   %12.4f   %12.4f   (%s)\n",
             m_stageMs[ALFP_STATS_CLASS_CPU] + m_stageMs[ALFP_STATS_CLASS_GPU], ( m_stageMs[ALFP_STATS_CLASS_CPU] + m_stageMs[ALFP_STATS_CLASS_GPU] ) / np,
             ( m_stageMs[ALFP_STATS_CLASS_CPU] + m_stageMs[ALFP_STATS_CLASS_GPU] ) / na, m_stageCalls[ALFP_STATS_CLASS_GPU] ? "GPU call + CPU for special CTUs" : "CPU" );
    fprintf( f, "    of which luma filtering [ms]         %10.3f   %12.4f   %12.4f   (%s)\n",
             m_stageMs[ALFP_RECON_LUMA_CPU] + m_stageMs[ALFP_RECON_LUMA_GPU], ( m_stageMs[ALFP_RECON_LUMA_CPU] + m_stageMs[ALFP_RECON_LUMA_GPU] ) / np,
             ( m_stageMs[ALFP_RECON_LUMA_CPU] + m_stageMs[ALFP_RECON_LUMA_GPU] ) / na, m_stageCalls[ALFP_RECON_LUMA_GPU] ? "GPU call + CPU for the rest" : "CPU" );
    const double covMs = m_stageMs[ALFP_STATS_COV_CPU] + m_stageMs[ALFP_STATS_COV_GPU];
    fprintf( f, "    of which covariance statistics [ms]  %10.3f   %12.4f   %12.4f   (%s)\n", covMs, covMs / np, covMs / na,
             m_stageCalls[ALFP_STATS_COV_GPU] ? "GPU call + CPU for special units" : "CPU" );
    const double other = alfMs - ( m_stageMs[ALFP_STATS_CLASS_CPU] + m_stageMs[ALFP_STATS_CLASS_GPU] + m_stageMs[ALFP_RECON_LUMA_CPU] + m_stageMs[ALFP_RECON_LUMA_GPU] + covMs );
    fprintf( f, "    rest [ms]                            %10.3f   %12.4f   %12.4f   (CPU: Wiener derivation, decisions, APS, chroma filter, CCALF, ...)\n", other, other / np, other / na );
    if( m_rapl.ok() )
    {
      const double dyn = m_cpuIdleW > 0 ? cpuAlfJ - m_cpuIdleW * alfMs * 1e-3 : -1;
      fprintf( f, "  CPU energy, RAPL in ALF scopes [J]     %10.4f   %12.5f   %12.5f\n", cpuAlfJ, cpuAlfJ / np, cpuAlfJ / na );
      if( dyn >= 0 ) fprintf( f, "  CPU energy above idle (%.2f W) [J]   %10.4f   %12.5f   %12.5f\n", m_cpuIdleW, dyn, dyn / np, dyn / na );
    }
    else
    {
      fprintf( f, "  CPU energy: RAPL not available (Linux powercap only)\n" );
    }
    if( gpuE )
    {
      fprintf( f, "  GPU energy while ALF active, NVML [J]  %10.4f   %12.5f   %12.5f\n", m_gpuAlfJ, m_gpuAlfJ / np, m_gpuAlfJ / na );
      if( m_gpuIdleW > 0 )
      {
        const double dyn = m_gpuAlfJ - m_gpuIdleW * activeS;
        fprintf( f, "  GPU energy above idle (%.2f W) [J]    %10.4f   %12.5f   %12.5f\n", m_gpuIdleW, dyn, dyn / np, dyn / na );
      }
      if( m_rapl.ok() )
        fprintf( f, "  CPU + GPU energy of the ALF [J]        %10.4f   %12.5f   %12.5f\n", cpuAlfJ + m_gpuAlfJ, ( cpuAlfJ + m_gpuAlfJ ) / np, ( cpuAlfJ + m_gpuAlfJ ) / na );
    }
  }

  if( m_gpuCalls )
  {
    const double n = ( double ) m_gpuCalls;
    fprintf( f, "\nGPU backend breakdown (%" PRIu64 " calls: classification, covariance, filter)  sum [ms]   avg [ms]\n", m_gpuCalls );
    fprintf( f, "  host pack (pageable->pinned)          %10.3f %10.4f\n", m_gpu.hostPackMs, m_gpu.hostPackMs / n );
    fprintf( f, "  H2D transfer                          %10.3f %10.4f   (%.1f MB, %.2f GB/s)\n", m_gpu.h2dMs, m_gpu.h2dMs / n,
             m_gpu.bytesH2D / 1e6, m_gpu.h2dMs > 0 ? m_gpu.bytesH2D / ( m_gpu.h2dMs * 1e6 ) : 0.0 );
    fprintf( f, "  classification kernel                 %10.3f %10.4f\n", m_gpu.classifyMs, m_gpu.classifyMs / n );
    fprintf( f, "  filter kernel                         %10.3f %10.4f\n", m_gpu.filterMs, m_gpu.filterMs / n );
    fprintf( f, "  covariance kernels                    %10.3f %10.4f\n", m_gpu.covMs, m_gpu.covMs / n );
    fprintf( f, "  D2H transfer                          %10.3f %10.4f   (%.1f MB, %.2f GB/s)\n", m_gpu.d2hMs, m_gpu.d2hMs / n,
             m_gpu.bytesD2H / 1e6, m_gpu.d2hMs > 0 ? m_gpu.bytesD2H / ( m_gpu.d2hMs * 1e6 ) : 0.0 );
    fprintf( f, "  host unpack (pinned->picture)         %10.3f %10.4f\n", m_gpu.hostUnpackMs, m_gpu.hostUnpackMs / n );
    fprintf( f, "  communication (H2D+D2H)               %10.3f\n", m_gpu.h2dMs + m_gpu.d2hMs );
    fprintf( f, "  execution (kernels)                   %10.3f\n", m_gpu.classifyMs + m_gpu.filterMs + m_gpu.covMs );
    fprintf( f, "  GPU timeline, overlapped streams      %10.3f %10.4f   (%.1f chunks/call; 1st H2D -> last D2H)\n", m_gpu.gpuSpanMs,
             m_gpu.gpuSpanMs / n, m_gpu.chunks / n );
    fprintf( f, "  total per call (wall, incl. host)     %10.3f %10.4f\n", m_gpu.totalMs, m_gpu.totalMs / n );
  }

  fprintf( f, "\nEnergy\n" );
  if( m_rapl.ok() )
  {
    const double cpuJ = ( m_cpuUJ1 - m_cpuUJ0 ) * 1e-6;
    fprintf( f, "  CPU package (RAPL) during encoding     %10.3f J  (avg %.2f W, idle baseline %.2f W)\n", cpuJ, cpuJ / encS, m_cpuIdleW );
    const double scopeJ = ( m_stageUJ[ALFP_STATS] + m_stageUJ[ALFP_DERIVE] + m_stageUJ[ALFP_RECON]
                          + m_stageUJ[ALFP_CCALF_STATS] + m_stageUJ[ALFP_CCALF_DERIVE] + m_stageUJ[ALFP_CCALF_RECON] ) * 1e-6;
    fprintf( f, "  CPU energy inside ALF scopes           %10.3f J  (valid with 1 thread; RAPL updates every ~1 ms)\n", scopeJ );
    fprintf( f, "  CPU energy x ALF time share (estimate) %10.3f J\n", cpuJ * alfMs / encMs );
  }
  else
  {
    fprintf( f, "  CPU: RAPL not available\n" );
  }
  if( m_backend && ( m_gpuEnergyOk || m_gpuSampledJ > 0 ) )
  {
    const double gpuS = std::chrono::duration<double>( m_t1Real - m_gpuT0 ).count();
    if( m_gpuEnergyOk ) fprintf( f, "  GPU (NVML energy counter)              %10.3f J  over %.3f s\n", ( m_gpuE1 - m_gpuE0 ) * 1e-3, gpuS );
    if( m_gpuIdleW > 0 )
    {
      const double totJ = m_gpuEnergyOk ? ( m_gpuE1 - m_gpuE0 ) * 1e-3 : m_gpuSampledJ;
      fprintf( f, "  GPU idle baseline %.2f W -> dynamic    %10.3f J  (= total - idle x time)\n", m_gpuIdleW, totJ - m_gpuIdleW * gpuS );
    }
    fprintf( f, "  GPU (NVML power sampling, 2 ms)        %10.3f J  (avg %.2f W, peak %.2f W)\n", m_gpuSampledJ, m_gpuSampledJ / encS, m_gpuPeakW );
    fprintf( f, "  note: GPU energy includes its idle power over the whole encode; subtract idle_W x time for the dynamic part\n" );
  }
  fprintf( f, "=============================================\n\n" );
}

} // namespace vvenc
