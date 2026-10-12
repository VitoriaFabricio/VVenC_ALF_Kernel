/* -----------------------------------------------------------------------------
alf_bench.cu - standalone, bit-exact ALF luma benchmark (CPU reference vs CUDA)

Replaces the original single-file prototype. It uses exactly the same core
(AlfGpuCore.h) and kernels (AlfGpuKernels.cuh) as the VVenC integration.

Inputs
  --dump <alf_pocNNNNN.bin>    frame dumped by VVenC (env VVENC_ALF_DUMP=<dir>): pre-ALF luma,
                               per-CTU on/off + filter set, APS sets, VVenC classification
  --csv  <frame.csv> --width W --height H [--ctu 128] [--bitdepth 10] [--set 0]
                               a luma frame as CSV; every CTU filtered with fixed filter set --set

Measurements (per frame, statistics over --iters iterations after --warmup)
  CPU : classification, filter, total (1 thread, same core as the GPU) + RAPL package energy
  GPU : H2D, classification kernel, filter kernel, D2H (CUDA events), wall total,
        "communication" (H2D+D2H), "execution" (kernels) + NVML energy (counter, minus idle)
  Bit-exactness: GPU output and classes vs CPU reference (and vs VVenC classes for --dump)
  Encoder pipeline: the same frame through the VVenC backend (AlfGpuCuda.cu), i.e. exactly the call
        made by --AlfGpu in frame mode: chunks of CTU rows pipelined over several CUDA streams
        (env VVENC_ALF_GPU_STREAMS, VVENC_ALF_GPU_CHUNK_ROWS). Without a GPU it runs the host emulation.

Build:  see test/alfgpu/Makefile
------------------------------------------------------------------------------- */

#include "../../source/Lib/CommonLib/AlfGpu.h"
#include "../../source/Lib/CommonLib/cuda/AlfGpuKernels.cuh"

#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if defined( __linux__ )
#  include <fcntl.h>
#  include <unistd.h>
#endif

using namespace alfgpu;
using clk = std::chrono::steady_clock;

#define CK( x )                                                                                          \
  do {                                                                                                   \
    cudaError_t e_ = ( x );                                                                              \
    if( e_ != cudaSuccess ) { fprintf( stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString( e_ ), __FILE__, __LINE__ ); exit( 1 ); } \
  } while( 0 )

static double msSince( clk::time_point a ) { return std::chrono::duration<double, std::milli>( clk::now() - a ).count(); }

// ---------------------------------------------------------------- RAPL (CPU package energy)
struct Rapl
{
  std::vector<int> fd; std::vector<uint64_t> maxr, last; uint64_t acc = 0;
  Rapl()
  {
#if defined( __linux__ )
    for( int i = 0; i < 16; i++ )
    {
      std::string b = "/sys/class/powercap/intel-rapl:" + std::to_string( i );
      int f = open( ( b + "/energy_uj" ).c_str(), O_RDONLY );
      if( f < 0 ) continue;
      char s[64] = { 0 };
      if( pread( f, s, 63, 0 ) <= 0 ) { close( f ); continue; }
      uint64_t m = 0;
      if( FILE* mf = fopen( ( b + "/max_energy_range_uj" ).c_str(), "r" ) ) { if( fscanf( mf, "%" SCNu64, &m ) != 1 ) m = 0; fclose( mf ); }
      fd.push_back( f ); maxr.push_back( m ); last.push_back( strtoull( s, nullptr, 10 ) );
    }
#endif
  }
  bool ok() const { return !fd.empty(); }
  double joules()
  {
#if defined( __linux__ )
    for( size_t i = 0; i < fd.size(); i++ )
    {
      char s[64] = { 0 };
      if( pread( fd[i], s, 63, 0 ) <= 0 ) continue;
      uint64_t v = strtoull( s, nullptr, 10 );
      acc += v >= last[i] ? v - last[i] : v + maxr[i] - last[i];
      last[i] = v;
    }
#endif
    return acc * 1e-6;
  }
};

// ---------------------------------------------------------------- NVML (GPU energy / power)
struct Nvml
{
  bool ok = false, energyOk = false; nvmlDevice_t dev{};
  void init( int cudaDev )
  {
    if( nvmlInit_v2() != NVML_SUCCESS ) return;
    char pci[32];
    if( cudaDeviceGetPCIBusId( pci, sizeof( pci ), cudaDev ) != cudaSuccess ) return;
    if( nvmlDeviceGetHandleByPciBusId_v2( pci, &dev ) != NVML_SUCCESS ) return;
    ok = true;
    unsigned long long e;
    energyOk = nvmlDeviceGetTotalEnergyConsumption( dev, &e ) == NVML_SUCCESS;
  }
  double joules() { unsigned long long e = 0; return ( energyOk && nvmlDeviceGetTotalEnergyConsumption( dev, &e ) == NVML_SUCCESS ) ? e * 1e-3 : -1; }
  double watts()  { unsigned p = 0; return ( ok && nvmlDeviceGetPowerUsage( dev, &p ) == NVML_SUCCESS ) ? p * 1e-3 : -1; }
};

// ---------------------------------------------------------------- frame
struct Frame
{
  int W = 0, H = 0, ctu = 128, bitDepth = 10, vbH = 128, vbPos = 124, numCols = 0, numRows = 0, poc = -1, numAps = 0;
  int pitch = 0;                   // W + 2*kHalo
  std::vector<Pel> pix;            // (H + 2*kHalo) x pitch, starts at (-kHalo, -kHalo)
  std::vector<int16_t> ctuSet;     // numCols*numRows
  std::vector<int16_t> coef, clip; // kMaxSets * kSetStride
  std::vector<uint8_t> vvencCls;   // optional
  const Pel* at0() const { return &pix[( size_t ) kHalo * pitch + kHalo]; }
};

// fixed filter tables (identical to VVenC AdaptiveLoopFilter::m_fixedFilterSetCoeff / m_classToFilterMapping)
static const int16_t kFixedCoeff[64][13] = {
  { 0, 0, 2, -3, 1, -4, 1, 7, -1, 1, -1, 5, 0 }, { 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, -1, 2, 0 }, { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 },
  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0 }, { 2, 2, -7, -3, 0, -5, 13, 22, 12, -3, -3, 17, 0 }, { -1, 0, 6, -8, 1, -5, 1, 23, 0, 2, -5, 10, 0 },
  { 0, 0, -1, -1, 0, -1, 2, 1, 0, 0, -1, 4, 0 }, { 0, 0, 3, -11, 1, 0, -1, 35, 5, 2, -9, 9, 0 }, { 0, 0, 8, -8, -2, -7, 4, 4, 2, 1, -1, 25, 0 },
  { 0, 0, 1, -1, 0, -3, 1, 3, -1, 1, -1, 3, 0 }, { 0, 0, 3, -3, 0, -6, 5, -1, 2, 1, -4, 21, 0 }, { -7, 1, 5, 4, -3, 5, 11, 13, 12, -8, 11, 12, 0 },
  { -5, -3, 6, -2, -3, 8, 14, 15, 2, -7, 11, 16, 0 }, { 2, -1, -6, -5, -2, -2, 20, 14, -4, 0, -3, 25, 0 }, { 3, 1, -8, -4, 0, -8, 22, 5, -3, 2, -10, 29, 0 },
  { 2, 1, -7, -1, 2, -11, 23, -5, 0, 2, -10, 29, 0 }, { -6, -3, 8, 9, -4, 8, 9, 7, 14, -2, 8, 9, 0 }, { 2, 1, -4, -7, 0, -8, 17, 22, 1, -1, -4, 23, 0 },
  { 3, 0, -5, -7, 0, -7, 15, 18, -5, 0, -5, 27, 0 }, { 2, 0, 0, -7, 1, -10, 13, 13, -4, 2, -7, 24, 0 }, { 3, 3, -13, 4, -2, -5, 9, 21, 25, -2, -3, 12, 0 },
  { -5, -2, 7, -3, -7, 9, 8, 9, 16, -2, 15, 12, 0 }, { 0, -1, 0, -7, -5, 4, 11, 11, 8, -6, 12, 21, 0 }, { 3, -2, -3, -8, -4, -1, 16, 15, -2, -3, 3, 26, 0 },
  { 2, 1, -5, -4, -1, -8, 16, 4, -2, 1, -7, 33, 0 }, { 2, 1, -4, -2, 1, -10, 17, -2, 0, 2, -11, 33, 0 }, { 1, -2, 7, -15, -16, 10, 8, 8, 20, 11, 14, 11, 0 },
  { 2, 2, 3, -13, -13, 4, 8, 12, 2, -3, 16, 24, 0 }, { 1, 4, 0, -7, -8, -4, 9, 9, -2, -2, 8, 29, 0 }, { 1, 1, 2, -4, -1, -6, 6, 3, -1, -1, -3, 30, 0 },
  { -7, 3, 2, 10, -2, 3, 7, 11, 19, -7, 8, 10, 0 }, { 0, -2, -5, -3, -2, 4, 20, 15, -1, -3, -1, 22, 0 }, { 3, -1, -8, -4, -1, -4, 22, 8, -4, 2, -8, 28, 0 },
  { 0, 3, -14, 3, 0, 1, 19, 17, 8, -3, -7, 20, 0 }, { 0, 2, -1, -8, 3, -6, 5, 21, 1, 1, -9, 13, 0 }, { -4, -2, 8, 20, -2, 2, 3, 5, 21, 4, 6, 1, 0 },
  { 2, -2, -3, -9, -4, 2, 14, 16, 3, -6, 8, 24, 0 }, { 2, 1, 5, -16, -7, 2, 3, 11, 15, -3, 11, 22, 0 }, { 1, 2, 3, -11, -2, -5, 4, 8, 9, -3, -2, 26, 0 },
  { 0, -1, 10, -9, -1, -8, 2, 3, 4, 0, 0, 29, 0 }, { 1, 2, 0, -5, 1, -9, 9, 3, 0, 1, -7, 20, 0 }, { -2, 8, -6, -4, 3, -9, -8, 45, 14, 2, -13, 7, 0 },
  { 1, -1, 16, -19, -8, -4, -3, 2, 19, 0, 4, 30, 0 }, { 1, 1, -3, 0, 2, -11, 15, -5, 1, 2, -9, 24, 0 }, { 0, 1, -2, 0, 1, -4, 4, 0, 0, 1, -4, 7, 0 },
  { 0, 1, 2, -5, 1, -6, 4, 10, -2, 1, -4, 10, 0 }, { 3, 0, -3, -6, -2, -6, 14, 8, -1, -1, -3, 31, 0 }, { 0, 1, 0, -2, 1, -6, 5, 1, 0, 1, -5, 13, 0 },
  { 3, 1, 9, -19, -21, 9, 7, 6, 13, 5, 15, 21, 0 }, { 2, 4, 3, -12, -13, 1, 7, 8, 3, 0, 12, 26, 0 }, { 3, 1, -8, -2, 0, -6, 18, 2, -2, 3, -10, 23, 0 },
  { 1, 1, -4, -1, 1, -5, 8, 1, -1, 2, -5, 10, 0 }, { 0, 1, -1, 0, 0, -2, 2, 0, 0, 1, -2, 3, 0 }, { 1, 1, -2, -7, 1, -7, 14, 18, 0, 0, -7, 21, 0 },
  { 0, 1, 0, -2, 0, -7, 8, 1, -2, 0, -3, 24, 0 }, { 0, 1, 1, -2, 2, -10, 10, 0, -2, 1, -7, 23, 0 }, { 0, 2, 2, -11, 2, -4, -3, 39, 7, 1, -10, 9, 0 },
  { 1, 0, 13, -16, -5, -6, -1, 8, 6, 0, 6, 29, 0 }, { 1, 3, 1, -6, -4, -7, 9, 6, -3, -2, 3, 33, 0 }, { 4, 0, -17, -1, -1, 5, 26, 8, -2, 3, -15, 30, 0 },
  { 0, 1, -2, 0, 2, -8, 12, -6, 1, 1, -6, 16, 0 }, { 0, 0, 0, -1, 1, -4, 4, 0, 0, 0, -3, 11, 0 }, { 0, 1, 2, -8, 2, -6, 5, 15, 0, 2, -7, 9, 0 },
  { 1, -1, 12, -15, -7, -2, 3, 6, 6, -1, 7, 30, 0 } };

static const int kClassToFilter[16][25] = {
  { 8, 2, 2, 2, 3, 4, 53, 9, 9, 52, 4, 4, 5, 9, 2, 8, 10, 9, 1, 3, 39, 39, 10, 9, 52 },
  { 11, 12, 13, 14, 15, 30, 11, 17, 18, 19, 16, 20, 20, 4, 53, 21, 22, 23, 14, 25, 26, 26, 27, 28, 10 },
  { 16, 12, 31, 32, 14, 16, 30, 33, 53, 34, 35, 16, 20, 4, 7, 16, 21, 36, 18, 19, 21, 26, 37, 38, 39 },
  { 35, 11, 13, 14, 43, 35, 16, 4, 34, 62, 35, 35, 30, 56, 7, 35, 21, 38, 24, 40, 16, 21, 48, 57, 39 },
  { 11, 31, 32, 43, 44, 16, 4, 17, 34, 45, 30, 20, 20, 7, 5, 21, 22, 46, 40, 47, 26, 48, 63, 58, 10 },
  { 12, 13, 50, 51, 52, 11, 17, 53, 45, 9, 30, 4, 53, 19, 0, 22, 23, 25, 43, 44, 37, 27, 28, 10, 55 },
  { 30, 33, 62, 51, 44, 20, 41, 56, 34, 45, 20, 41, 41, 56, 5, 30, 56, 38, 40, 47, 11, 37, 42, 57, 8 },
  { 35, 11, 23, 32, 14, 35, 20, 4, 17, 18, 21, 20, 20, 20, 4, 16, 21, 36, 46, 25, 41, 26, 48, 49, 58 },
  { 12, 31, 59, 59, 3, 33, 33, 59, 59, 52, 4, 33, 17, 59, 55, 22, 36, 59, 59, 60, 22, 36, 59, 25, 55 },
  { 31, 25, 15, 60, 60, 22, 17, 19, 55, 55, 20, 20, 53, 19, 55, 22, 46, 25, 43, 60, 37, 28, 10, 55, 52 },
  { 12, 31, 32, 50, 51, 11, 33, 53, 19, 45, 16, 4, 4, 53, 5, 22, 36, 18, 25, 43, 26, 27, 27, 28, 10 },
  { 5, 2, 44, 52, 3, 4, 53, 45, 9, 3, 4, 56, 5, 0, 2, 5, 10, 47, 52, 3, 63, 39, 10, 9, 52 },
  { 12, 34, 44, 44, 3, 56, 56, 62, 45, 9, 56, 56, 7, 5, 0, 22, 38, 40, 47, 52, 48, 57, 39, 10, 9 },
  { 35, 11, 23, 14, 51, 35, 20, 41, 56, 62, 16, 20, 41, 56, 7, 16, 21, 38, 24, 40, 26, 26, 42, 57, 39 },
  { 33, 34, 51, 51, 52, 41, 41, 34, 62, 0, 41, 41, 56, 7, 5, 56, 38, 38, 40, 44, 37, 42, 57, 39, 10 },
  { 16, 31, 32, 15, 60, 30, 4, 17, 19, 25, 22, 20, 4, 53, 19, 21, 22, 46, 25, 55, 26, 48, 63, 58, 55 } };

static void buildFixedTables( Frame& f )
{
  f.coef.assign( kMaxSets * kSetStride, 0 );
  f.clip.assign( kMaxSets * kSetStride, 0 );
  for( int s = 0; s < kNumFixedSets; s++ )
    for( int c = 0; c < kNumClasses; c++ )
    {
      for( int k = 0; k < 12; k++ ) f.coef[s * kSetStride + c * kNumCoeff + k] = kFixedCoeff[kClassToFilter[s][c]][k];
      f.coef[s * kSetStride + c * kNumCoeff + 12] = 1 << kShift;            // centre coefficient 128
      for( int k = 0; k < 13; k++ ) f.clip[s * kSetStride + c * kNumCoeff + k] = ( int16_t ) ( 1 << f.bitDepth );   // m_clipDefault
    }
}

static bool loadDump( const char* path, Frame& f )
{
  FILE* fp = fopen( path, "rb" );
  if( !fp ) return false;
  char magic[4];
  int32_t h[12];
  bool ok = fread( magic, 1, 4, fp ) == 4 && !memcmp( magic, "ALFD", 4 ) && fread( h, sizeof( h ), 1, fp ) == 1 && h[0] == 1;
  if( !ok ) { fclose( fp ); return false; }
  f.W = h[1]; f.H = h[2]; f.ctu = h[3]; f.bitDepth = h[4]; f.vbH = h[5]; f.vbPos = h[6];
  f.numCols = h[7]; f.numRows = h[8]; f.poc = h[10]; f.numAps = h[11];
  if( h[9] != kHalo ) { fclose( fp ); fprintf( stderr, "unexpected halo\n" ); return false; }
  f.pitch = f.W + 2 * kHalo;
  f.pix.resize( ( size_t ) f.pitch * ( f.H + 2 * kHalo ) );
  f.ctuSet.resize( f.numCols * f.numRows );
  f.coef.assign( kMaxSets * kSetStride, 0 ); f.clip.assign( kMaxSets * kSetStride, 0 );
  f.vvencCls.resize( ( size_t ) ( f.W >> 2 ) * ( ( f.H + 3 ) >> 2 ) );
  const size_t na = ( size_t ) f.numAps * kSetStride, nf = ( size_t ) kNumFixedSets * kSetStride;
  ok = fread( f.pix.data(), sizeof( Pel ), f.pix.size(), fp ) == f.pix.size()
    && fread( f.ctuSet.data(), sizeof( int16_t ), f.ctuSet.size(), fp ) == f.ctuSet.size()
    && fread( &f.coef[nf], sizeof( int16_t ), na, fp ) == na
    && fread( &f.clip[nf], sizeof( int16_t ), na, fp ) == na
    && fread( &f.coef[0], sizeof( int16_t ), nf, fp ) == nf
    && fread( &f.clip[0], sizeof( int16_t ), nf, fp ) == nf
    && fread( f.vvencCls.data(), 1, f.vvencCls.size(), fp ) == f.vvencCls.size();
  fclose( fp );
  return ok;
}

static bool loadCsv( const char* path, int W, int H, int ctu, int bitDepth, int set, Frame& f )
{
  FILE* fp = fopen( path, "r" );
  if( !fp ) return false;
  f.W = W; f.H = H; f.ctu = ctu; f.bitDepth = bitDepth; f.vbH = ctu; f.vbPos = ctu - 4;
  f.numCols = ( W + ctu - 1 ) / ctu; f.numRows = ( H + ctu - 1 ) / ctu;
  f.pitch = W + 2 * kHalo;
  f.pix.assign( ( size_t ) f.pitch * ( H + 2 * kHalo ), 0 );
  Pel* p = &f.pix[( size_t ) kHalo * f.pitch + kHalo];
  const int maxVal = ( 1 << bitDepth ) - 1;
  for( int i = 0; i < W * H; i++ )
  {
    int v;
    if( fscanf( fp, " %d ,", &v ) != 1 ) { fprintf( stderr, "CSV: only %d of %d samples\n", i, W * H ); fclose( fp ); return false; }
    p[( i / W ) * f.pitch + i % W] = ( Pel ) std::max( 0, std::min( maxVal, v ) );
  }
  fclose( fp );
  // replicate padding of kHalo samples (what VVenC's extendBorderPel does at picture borders)
  for( int y = 0; y < H; y++ )
    for( int k = 1; k <= kHalo; k++ ) { p[y * f.pitch - k] = p[y * f.pitch]; p[y * f.pitch + W - 1 + k] = p[y * f.pitch + W - 1]; }
  for( int k = 1; k <= kHalo; k++ )
  {
    memcpy( p - ( ptrdiff_t ) k * f.pitch - kHalo, p - kHalo, f.pitch * sizeof( Pel ) );
    memcpy( p + ( ptrdiff_t ) ( H - 1 + k ) * f.pitch - kHalo, p + ( ptrdiff_t ) ( H - 1 ) * f.pitch - kHalo, f.pitch * sizeof( Pel ) );
  }
  f.ctuSet.assign( f.numCols * f.numRows, ( int16_t ) set );
  buildFixedTables( f );
  return true;
}

struct Stat
{
  std::vector<double> v;
  void add( double x ) { v.push_back( x ); }
  double mean() const { double s = 0; for( double x : v ) s += x; return v.empty() ? 0 : s / v.size(); }
  double sd() const { double m = mean(), s = 0; for( double x : v ) s += ( x - m ) * ( x - m ); return v.size() > 1 ? std::sqrt( s / ( v.size() - 1 ) ) : 0; }
  double min() const { return v.empty() ? 0 : *std::min_element( v.begin(), v.end() ); }
  double med() const { if( v.empty() ) return 0; std::vector<double> c( v ); std::nth_element( c.begin(), c.begin() + c.size() / 2, c.end() ); return c[c.size() / 2]; }
};

static void printStat( const char* n, const Stat& s )
{
  printf( "  %-34s mean %9.4f  sd %8.4f  min %9.4f  median %9.4f ms\n", n, s.mean(), s.sd(), s.min(), s.med() );
}

// ---------------------------------------------------------------- encoder backend (frame mode pipeline)
static void runBackendPipeline( const Frame& f, const BandParams& bp0, const std::vector<Pel>& refOut, const std::vector<uint8_t>& refCls,
                                bool cpuClass, int iters, int warmup, double minSeconds, double cpuMs, Stat& sSpan, Stat& sCall, bool& exact )
{
  Backend* be = createBackend( &f.coef[0], &f.clip[0], f.W, f.H, 1 );
  BandParams bp = bp0;
  bp.numAps   = f.numAps;
  bp.apsCoef  = &f.coef[kNumFixedSets * kSetStride];
  bp.apsClip  = &f.clip[kNumFixedSets * kSetStride];
  bp.cpuClass = cpuClass ? refCls.data() : nullptr;
  const int clsStride = f.W >> 2;
  std::vector<Pel> out( ( size_t ) f.W * f.H, 0 );
  std::vector<uint8_t> cls( ( size_t ) clsStride * ( ( f.H + 3 ) >> 2 ), 255 );
  bp.dst = out.data(); bp.dstStride = f.W; bp.outClass = cls.data();
  // --AlfGpu 1 path: classifyFrame (start of the ALF: upload + gradients/classes/transpose + classes back)
  //                  + filterFrameResident (after the decisions: map/APS up, filter, picture back)
  // --cpu-class   : --AlfGpu 2 path (one call: upload picture + CPU classes, filter)
  const int mode = cpuClass ? MODE_FILTER : MODE_FULL;
  const int slot = be->acquireSlot();
  std::vector<int16_t> clsMap( f.ctuSet.size(), 0 );          // classification: every CTU
  BandParams bc = bp; bc.ctuSetIdx = clsMap.data(); bc.numAps = 0; bc.dst = nullptr; bc.outClass = cls.data();
  BandParams bf = bp; bf.outClass = nullptr;
  Timing t;
  auto once = [&]( Timing& tm )
  {
    if( cpuClass ) be->filterLumaBand( bp, mode, 0, tm );
    else { be->classifyFrame( bc, slot, tm ); be->filterFrameResident( bf, slot, tm ); }
  };
  for( int i = 0; i < warmup; i++ ) { t = Timing(); once( t ); }
  Stat sPack, sH2D, sCls, sFlt, sD2H, sUnp;
  int n = 0; const auto tStart = clk::now();
  while( true )
  {
    t = Timing();
    once( t );
    sPack.add( t.hostPackMs ); sH2D.add( t.h2dMs ); sCls.add( t.classifyMs ); sFlt.add( t.filterMs ); sD2H.add( t.d2hMs );
    sUnp.add( t.hostUnpackMs ); sSpan.add( t.gpuSpanMs ); sCall.add( t.totalMs ); n++;
    if( ( iters > 0 && n >= iters ) || ( iters <= 0 && msSince( tStart ) > minSeconds * 1e3 && n >= 10 ) ) break;
  }
  const char* st = getenv( "VVENC_ALF_GPU_STREAMS" );
  printf( "\nEncoder backend '%s', frame mode (%s), %d chunks, %s streams (%d iterations)\n", be->name(),
          cpuClass ? "--AlfGpu 2: one call" : "--AlfGpu 1: classifyFrame + filterFrameResident", t.chunks, st ? st : "default", n );
  printStat( "host pack (picture -> pinned)", sPack );
  printStat( "H2D (sum over chunks)", sH2D );
  printStat( "classification kernel (sum)", sCls );
  printStat( "filter kernel (sum)", sFlt );
  printStat( "D2H (sum)", sD2H );
  printStat( "host unpack (pinned -> picture)", sUnp );
  printStat( "GPU timeline, overlapped", sSpan );
  printStat( "total GPU calls (wall, as in VVenC)", sCall );
  if( !strcmp( be->name(), "host-emu" ) )
    printf( "  (host emulation: the kernels are simulated on the CPU, these times are NOT GPU times)\n" );
  else
    printf( "  speed-up vs CPU reference (end-to-end call): %.2fx\n", cpuMs / sCall.mean() );

  size_t bad = 0;
  for( int y = 0; y < f.H; y++ )
    for( int x = 0; x < f.W; x++ )
    {
      if( f.ctuSet[( y / f.ctu ) * f.numCols + x / f.ctu] < 0 ) continue;
      bad += out[( size_t ) y * f.W + x] != refOut[( size_t ) y * f.W + x];
      if( !cpuClass && ( x & 3 ) == 0 && ( y & 3 ) == 0 ) bad += cls[( y >> 2 ) * clsStride + ( x >> 2 )] != refCls[( y >> 2 ) * clsStride + ( x >> 2 )];
    }
  exact = bad == 0;
  printf( "  bit-exactness (samples + classes): %zu mismatches -> %s\n", bad, exact ? "BIT-EXACT" : "NOT BIT-EXACT" );
  delete be;
}

static void usage()
{
  printf( "usage: alf_bench (--dump file.bin | --csv file.csv --width W --height H [--ctu 128] [--bitdepth 10] [--set 0])\n"
          "                 [--iters N] [--warmup 5] [--min-seconds 2] [--pageable] [--cpu-class] [--device 0]\n"
          "                 [--idle-ms 1000] [--out filtered.csv] [--classes classes.csv] [--label name] [--csvlog results.csv]\n" );
}

int main( int argc, char** argv )
{
  const char *dump = nullptr, *csv = nullptr, *outCsv = nullptr, *clsCsv = nullptr, *label = "", *csvlog = nullptr;
  int W = 0, H = 0, ctu = 128, bitDepth = 10, set = 0, iters = 0, warmup = 5, device = 0, idleMs = 1000;
  double minSeconds = 2.0;
  bool pinned = true, cpuClass = false;
  for( int i = 1; i < argc; i++ )
  {
    std::string a = argv[i];
    auto nxt = [&]() { if( i + 1 >= argc ) { usage(); exit( 1 ); } return argv[++i]; };
    if( a == "--dump" ) dump = nxt();
    else if( a == "--csv" ) csv = nxt();
    else if( a == "--width" ) W = atoi( nxt() );
    else if( a == "--height" ) H = atoi( nxt() );
    else if( a == "--ctu" ) ctu = atoi( nxt() );
    else if( a == "--bitdepth" ) bitDepth = atoi( nxt() );
    else if( a == "--set" ) set = atoi( nxt() );
    else if( a == "--iters" ) iters = atoi( nxt() );
    else if( a == "--warmup" ) warmup = atoi( nxt() );
    else if( a == "--min-seconds" ) minSeconds = atof( nxt() );
    else if( a == "--pageable" ) pinned = false;
    else if( a == "--cpu-class" ) cpuClass = true;
    else if( a == "--device" ) device = atoi( nxt() );
    else if( a == "--idle-ms" ) idleMs = atoi( nxt() );
    else if( a == "--out" ) outCsv = nxt();
    else if( a == "--classes" ) clsCsv = nxt();
    else if( a == "--label" ) label = nxt();
    else if( a == "--csvlog" ) csvlog = nxt();
    else { usage(); return 1; }
  }

  Frame f;
  if( dump ) { if( !loadDump( dump, f ) ) { fprintf( stderr, "cannot read dump %s\n", dump ); return 1; } }
  else if( csv && W > 0 && H > 0 ) { if( !loadCsv( csv, W, H, ctu, bitDepth, set, f ) ) { fprintf( stderr, "cannot read csv %s\n", csv ); return 1; } }
  else { usage(); return 1; }

  int numFiltered = 0;
  for( int16_t s : f.ctuSet ) numFiltered += s >= 0;
  printf( "frame %dx%d, %d-bit, CTU %d, VB at %d, %d/%d CTUs filtered, %d APS sets%s\n", f.W, f.H, f.bitDepth, f.ctu, f.vbPos,
          numFiltered, f.numCols * f.numRows, f.numAps, dump ? " (VVenC dump)" : " (CSV)" );

  // the whole frame is one band (bandY0 = 0): identical results to VVenC's CTU-row bands because
  // all VB / boundary decisions use global coordinates
  BandParams bp{};
  bp.src = f.at0(); bp.srcStride = f.pitch; bp.picWidth = f.W; bp.bandY0 = 0; bp.bandH = f.H; bp.ctuSize = f.ctu;
  bp.numCtuCols = f.numCols; bp.bitDepth = f.bitDepth; bp.vbH = f.vbH; bp.vbPos = f.vbPos; bp.ctuSetIdx = f.ctuSet.data();
  const int clsStride = f.W >> 2, clsRows = ( f.H + 3 ) >> 2;
  const size_t outN = ( size_t ) f.W * f.H, clsN = ( size_t ) clsStride * clsRows, inN = f.pix.size();

  // ------------------------------------------------------------------ CPU reference (1 thread)
  std::vector<uint8_t> refCls( clsN, 255 );
  std::vector<Pel> refOut( outN, 0 );
  Rapl rapl;
  Stat cpuCls, cpuFlt, cpuTot;
  {
    // CPU idle baseline
    double cpuIdleW = -1;
    if( rapl.ok() && idleMs > 0 )
    {
      double e0 = rapl.joules(); auto t0 = clk::now();
      std::this_thread::sleep_for( std::chrono::milliseconds( idleMs ) );
      cpuIdleW = ( rapl.joules() - e0 ) / ( msSince( t0 ) * 1e-3 );
    }
    int n = 0; const auto tStart = clk::now(); const double e0 = rapl.ok() ? rapl.joules() : 0;
    while( true )
    {
      auto t0 = clk::now();
      hostClassifyBand( f.at0(), f.pitch, bp, refCls.data(), clsStride );
      const double a = msSince( t0 ); auto t1 = clk::now();
      hostFilterBand( f.at0(), f.pitch, bp, f.coef.data(), f.clip.data(), refCls.data(), clsStride, refOut.data(), f.W );
      const double b = msSince( t1 );
      cpuCls.add( a ); cpuFlt.add( b ); cpuTot.add( a + b ); n++;
      if( ( iters > 0 && n >= iters ) || ( iters <= 0 && msSince( tStart ) > minSeconds * 1e3 && n >= 3 ) ) break;
    }
    const double s = msSince( tStart ) * 1e-3, e = rapl.ok() ? rapl.joules() - e0 : -1;
    printf( "\nCPU reference (1 thread, shared core, %d iterations)\n", n );
    printStat( "classification", cpuCls ); printStat( "filter", cpuFlt ); printStat( "total", cpuTot );
    if( rapl.ok() )
      printf( "  energy (RAPL package): %.4f J/frame total, idle %.2f W -> dynamic %.4f J/frame\n", e / n, cpuIdleW, ( e - cpuIdleW * s ) / n );
    else
      printf( "  energy: RAPL not readable (run as root or chmod o+r /sys/class/powercap/intel-rapl:*/energy_uj)\n" );
  }
  if( !f.vvencCls.empty() )
  {
    size_t bad = 0, cnt = 0;
    for( size_t i = 0; i < clsN; i++ ) if( f.vvencCls[i] != 255 ) { cnt++; bad += f.vvencCls[i] != refCls[i]; }
    printf( "  classification vs VVenC (dump): %zu blocks, %zu mismatches\n", cnt, bad );
  }

  auto writeOutputs = [&]( const Pel* out, const uint8_t* cls )
  {
    if( outCsv )
    {
      if( FILE* o = fopen( outCsv, "w" ) )
      {
        for( int y = 0; y < f.H; y++ )
        {
          for( int x = 0; x < f.W; x++ )
          {
            const bool filt = f.ctuSet[( y / f.ctu ) * f.numCols + x / f.ctu] >= 0;
            fprintf( o, x ? ",%d" : "%d", filt ? out[( size_t ) y * f.W + x] : f.at0()[( size_t ) y * f.pitch + x] );
          }
          fprintf( o, "\n" );
        }
        fclose( o );
      }
    }
    if( clsCsv )
    {
      if( FILE* o = fopen( clsCsv, "w" ) )
      {
        for( int y = 0; y < clsRows; y++ )
        {
          for( int x = 0; x < clsStride; x++ ) fprintf( o, x ? ",%d" : "%d", cls[y * clsStride + x] == 255 ? -1 : cls[y * clsStride + x] & 31 );
          fprintf( o, "\n" );
        }
        fclose( o );
      }
    }
  };

  // ------------------------------------------------------------------ GPU
  int ndev = 0;
  if( cudaGetDeviceCount( &ndev ) != cudaSuccess || ndev == 0 )
  {
    printf( "\nno CUDA device - single-stream GPU part skipped (--out/--classes contain the CPU reference)\n" );
    Stat sSpan, sCall; bool ex = false;
    runBackendPipeline( f, bp, refOut, refCls, cpuClass, iters, 1, minSeconds, cpuTot.mean(), sSpan, sCall, ex );
    writeOutputs( refOut.data(), refCls.data() );
    return ex ? 0 : 2;
  }
  CK( cudaSetDevice( device ) );
  {
    char d[16]; snprintf( d, sizeof( d ), "%d", device );
    // the encoder backend uses the same device
#if defined( _WIN32 )
    if( !getenv( "VVENC_ALF_GPU_DEVICE" ) ) _putenv_s( "VVENC_ALF_GPU_DEVICE", d );
#else
    setenv( "VVENC_ALF_GPU_DEVICE", d, 0 );
#endif
  }
  cudaDeviceProp prop; CK( cudaGetDeviceProperties( &prop, device ) );
  printf( "\nGPU: %s (sm_%d%d), %s host buffers, classification on %s\n", prop.name, prop.major, prop.minor,
          pinned ? "pinned" : "pageable", cpuClass ? "CPU (uploaded)" : "GPU" );
  Nvml nvml; nvml.init( device );

  Pel *hIn, *hOut; uint8_t* hCls;
  if( pinned ) { CK( cudaMallocHost( &hIn, inN * sizeof( Pel ) ) ); CK( cudaMallocHost( &hOut, outN * sizeof( Pel ) ) ); CK( cudaMallocHost( &hCls, clsN ) ); }
  else { hIn = ( Pel* ) malloc( inN * sizeof( Pel ) ); hOut = ( Pel* ) malloc( outN * sizeof( Pel ) ); hCls = ( uint8_t* ) malloc( clsN ); }
  memcpy( hIn, f.pix.data(), inN * sizeof( Pel ) );
  if( cpuClass ) memcpy( hCls, refCls.data(), clsN );

  Pel *dIn, *dOut; uint8_t* dCls; int16_t *dMap, *dCoef, *dClip;
  CK( cudaMalloc( &dIn, inN * sizeof( Pel ) ) );   CK( cudaMalloc( &dOut, outN * sizeof( Pel ) ) );
  CK( cudaMalloc( &dCls, clsN ) );                 CK( cudaMalloc( &dMap, f.ctuSet.size() * sizeof( int16_t ) ) );
  CK( cudaMalloc( &dCoef, f.coef.size() * sizeof( int16_t ) ) ); CK( cudaMalloc( &dClip, f.clip.size() * sizeof( int16_t ) ) );
  CK( cudaMemset( dOut, 0, outN * sizeof( Pel ) ) );
  // fixed sets are resident; the APS sets (as in the encoder) are uploaded every frame
  CK( cudaMemcpy( dCoef, f.coef.data(), f.coef.size() * sizeof( int16_t ), cudaMemcpyHostToDevice ) );
  CK( cudaMemcpy( dClip, f.clip.data(), f.clip.size() * sizeof( int16_t ), cudaMemcpyHostToDevice ) );

  KParams k;
  k.picWidth = f.W; k.bandY0 = 0; k.bandH = f.H; k.ctuSize = f.ctu; k.numCtuCols = f.numCols; k.bitDepth = f.bitDepth;
  k.vbH = f.vbH; k.vbPos = f.vbPos; k.inPitch = f.pitch; k.clsStride = clsStride; k.outPitch = f.W;

  cudaStream_t s; CK( cudaStreamCreate( &s ) );
  cudaEvent_t ev[5]; for( auto& e : ev ) CK( cudaEventCreate( &e ) );
  const size_t apsN = ( size_t ) f.numAps * kSetStride;

  auto runOnce = [&]( double* seg, double& wall )
  {
    auto t0 = clk::now();
    CK( cudaEventRecord( ev[0], s ) );
    CK( cudaMemcpyAsync( dIn, hIn, inN * sizeof( Pel ), cudaMemcpyHostToDevice, s ) );
    CK( cudaMemcpyAsync( dMap, f.ctuSet.data(), f.ctuSet.size() * sizeof( int16_t ), cudaMemcpyHostToDevice, s ) );
    if( apsN )
    {
      CK( cudaMemcpyAsync( dCoef + kNumFixedSets * kSetStride, &f.coef[kNumFixedSets * kSetStride], apsN * sizeof( int16_t ), cudaMemcpyHostToDevice, s ) );
      CK( cudaMemcpyAsync( dClip + kNumFixedSets * kSetStride, &f.clip[kNumFixedSets * kSetStride], apsN * sizeof( int16_t ), cudaMemcpyHostToDevice, s ) );
    }
    if( cpuClass ) CK( cudaMemcpyAsync( dCls, hCls, clsN, cudaMemcpyHostToDevice, s ) );
    CK( cudaEventRecord( ev[1], s ) );
    if( !cpuClass ) launchAlfClassify( dIn, dMap, dCls, k, s );
    CK( cudaEventRecord( ev[2], s ) );
    launchAlfFilterLuma( dIn, dMap, dCls, dCoef, dClip, dOut, k, s );
    CK( cudaEventRecord( ev[3], s ) );
    CK( cudaMemcpyAsync( hOut, dOut, outN * sizeof( Pel ), cudaMemcpyDeviceToHost, s ) );
    CK( cudaEventRecord( ev[4], s ) );
    CK( cudaEventSynchronize( ev[4] ) );
    wall = msSince( t0 );
    float m;
    for( int i = 0; i < 4; i++ ) { CK( cudaEventElapsedTime( &m, ev[i], ev[i + 1] ) ); seg[i] = m; }
    CK( cudaGetLastError() );
  };

  double seg[4], wall;
  for( int i = 0; i < warmup; i++ ) runOnce( seg, wall );

  // GPU / CPU idle baselines just before the measurement
  double gpuIdleW = -1, cpuIdleW = -1;
  if( idleMs > 0 )
  {
    double acc = 0; int n = 0; const double e0 = rapl.ok() ? rapl.joules() : 0; auto t0 = clk::now();
    for( int t = 0; t < idleMs; t += 10 ) { std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) ); double w = nvml.watts(); if( w >= 0 ) { acc += w; n++; } }
    gpuIdleW = n ? acc / n : -1;
    if( rapl.ok() ) cpuIdleW = ( rapl.joules() - e0 ) / ( msSince( t0 ) * 1e-3 );
  }

  Stat sH2D, sCls, sFlt, sD2H, sComm, sExec, sWall;
  int n = 0; const auto tStart = clk::now();
  const double g0 = nvml.joules(), c0 = rapl.ok() ? rapl.joules() : 0;
  while( true )
  {
    runOnce( seg, wall );
    sH2D.add( seg[0] ); sCls.add( seg[1] ); sFlt.add( seg[2] ); sD2H.add( seg[3] );
    sComm.add( seg[0] + seg[3] ); sExec.add( seg[1] + seg[2] ); sWall.add( wall ); n++;
    if( ( iters > 0 && n >= iters ) || ( iters <= 0 && msSince( tStart ) > minSeconds * 1e3 && n >= 10 ) ) break;
  }
  const double secs = msSince( tStart ) * 1e-3;
  const double gJ = nvml.energyOk ? nvml.joules() - g0 : -1, cJ = rapl.ok() ? rapl.joules() - c0 : -1;

  printf( "GPU (%d iterations, CUDA events)\n", n );
  printStat( "H2D (input band + CTU map + APS)", sH2D );
  printStat( "classification kernel", sCls );
  printStat( "filter kernel", sFlt );
  printStat( "D2H (filtered frame)", sD2H );
  printStat( "communication (H2D + D2H)", sComm );
  printStat( "execution (kernels)", sExec );
  printStat( "total (wall clock, incl. sync)", sWall );
  printf( "  H2D %.2f MB -> %.2f GB/s, D2H %.2f MB -> %.2f GB/s\n", inN * 2e-6, inN * 2e-6 / sH2D.mean(), outN * 2e-6, outN * 2e-6 / sD2H.mean() );
  if( gJ >= 0 )
    printf( "  energy GPU (NVML counter): %.4f J/frame total, idle %.2f W -> dynamic %.4f J/frame\n", gJ / n, gpuIdleW, ( gJ - gpuIdleW * secs ) / n );
  else
    printf( "  energy GPU: NVML energy counter not supported on this GPU (needs Volta or newer)\n" );
  if( cJ >= 0 )
    printf( "  energy CPU during GPU run (RAPL): %.4f J/frame total, idle %.2f W -> dynamic %.4f J/frame\n", cJ / n, cpuIdleW, ( cJ - cpuIdleW * secs ) / n );
  printf( "  speed-up vs CPU reference: kernels only %.2fx, end-to-end %.2fx\n", cpuTot.mean() / sExec.mean(), cpuTot.mean() / sWall.mean() );

  // ------------------------------------------------------------------ bit-exactness
  std::vector<uint8_t> gpuCls( clsN );
  CK( cudaMemcpy( gpuCls.data(), dCls, clsN, cudaMemcpyDeviceToHost ) );
  size_t badPix = 0, nPix = 0, badCls = 0, nCls = 0;
  for( int y = 0; y < f.H; y++ )
    for( int x = 0; x < f.W; x++ )
    {
      if( f.ctuSet[( y / f.ctu ) * f.numCols + x / f.ctu] < 0 ) continue;
      nPix++;
      if( hOut[( size_t ) y * f.W + x] != refOut[( size_t ) y * f.W + x] && badPix++ < 5 )
        printf( "  mismatch (%d,%d): gpu %d cpu %d\n", x, y, hOut[( size_t ) y * f.W + x], refOut[( size_t ) y * f.W + x] );
      if( ( x & 3 ) == 0 && ( y & 3 ) == 0 ) { nCls++; badCls += gpuCls[( y >> 2 ) * clsStride + ( x >> 2 )] != refCls[( y >> 2 ) * clsStride + ( x >> 2 )]; }
    }
  printf( "\nbit-exactness: %zu samples / %zu mismatches, %zu classes / %zu mismatches -> %s\n", nPix, badPix, nCls, badCls,
          ( badPix || badCls ) ? "NOT BIT-EXACT" : "BIT-EXACT" );

  // ------------------------------------------------------------------ encoder backend (multi-stream frame pipeline)
  Stat pSpan, pCall; bool pExact = false;
  runBackendPipeline( f, bp, refOut, refCls, cpuClass, iters, warmup, minSeconds, cpuTot.mean(), pSpan, pCall, pExact );
  if( !pExact ) badPix++;

  // ------------------------------------------------------------------ outputs
  writeOutputs( hOut, gpuCls.data() );
  if( csvlog )
  {
    FILE* o = fopen( csvlog, "a" );
    if( o )
    {
      fseek( o, 0, SEEK_END );
      if( ftell( o ) == 0 )
        fprintf( o, "label,width,height,gpu,pinned,cpu_class,cpu_class_ms,cpu_filter_ms,cpu_total_ms,h2d_ms,class_ms,filter_ms,d2h_ms,comm_ms,exec_ms,wall_ms,gpu_J_frame,gpu_idle_W,cpu_J_frame_gpu_run,bitexact,pipe_span_ms,pipe_call_ms\n" );
      fprintf( o, "%s,%d,%d,%s,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.5f,%.2f,%.5f,%d,%.4f,%.4f\n", label, f.W, f.H, prop.name, pinned, cpuClass,
               cpuCls.mean(), cpuFlt.mean(), cpuTot.mean(), sH2D.mean(), sCls.mean(), sFlt.mean(), sD2H.mean(), sComm.mean(), sExec.mean(), sWall.mean(),
               gJ >= 0 ? gJ / n : -1.0, gpuIdleW, cJ >= 0 ? cJ / n : -1.0, ( badPix || badCls ) ? 0 : 1, pSpan.mean(), pCall.mean() );
      fclose( o );
    }
  }

  for( auto& e : ev ) cudaEventDestroy( e );
  cudaStreamDestroy( s );
  cudaFree( dIn ); cudaFree( dOut ); cudaFree( dCls ); cudaFree( dMap ); cudaFree( dCoef ); cudaFree( dClip );
  if( pinned ) { cudaFreeHost( hIn ); cudaFreeHost( hOut ); cudaFreeHost( hCls ); } else { free( hIn ); free( hOut ); free( hCls ); }
  if( nvml.ok ) nvmlShutdown();
  return ( badPix || badCls ) ? 2 : 0;
}
