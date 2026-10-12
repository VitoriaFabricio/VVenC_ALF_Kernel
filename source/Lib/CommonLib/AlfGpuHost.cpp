/* -----------------------------------------------------------------------------
AlfGpuHost.cpp  -  host reference loops + host-emulation backend

The emulation backend mimics the GPU data flow (pack band with halo -> classify ->
filter -> unpack) but runs the shared core on the CPU. It exists to prove that
the integration (indices, halos, VB, borders, per-CTU sets) is bit-exact with
VVenC on machines without a GPU, and as a CPU baseline for the benchmark.
------------------------------------------------------------------------------- */

#include "AlfGpu.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mutex>
#include <memory>

namespace alfgpu
{

static inline int ctuOf( const BandParams& p, int x, int yInBand )
{
  return ( yInBand / p.ctuSize ) * p.numCtuCols + x / p.ctuSize;
}

void hostClassifyBand( const Pel* in, ptrdiff_t inStride, const BandParams& p, uint8_t* cls, int clsStride )
{
  const int shift = p.bitDepth + 4;
  for( int yb = 0; yb < p.bandH; yb += 4 )
  {
    for( int x = 0; x < p.picWidth; x += 4 )
    {
      if( p.ctuSetIdx[ctuOf( p, x, yb )] < 0 ) continue;
      int c, t;
      classifyBlock4x4( in + yb * inStride + x, inStride, p.bandY0 + yb, shift, p.vbH, p.vbPos, c, t );
      cls[( yb >> 2 ) * clsStride + ( x >> 2 )] = ( uint8_t ) ( c | ( t << 5 ) );
    }
  }
}

void hostFilterBand( const Pel* in, ptrdiff_t inStride, const BandParams& p, const int16_t* coefTab, const int16_t* clipTab,
                     const uint8_t* cls, int clsStride, Pel* out, ptrdiff_t outStride )
{
  const int maxVal = ( 1 << p.bitDepth ) - 1;
  for( int yb = 0; yb < p.bandH; yb++ )
  {
    for( int x = 0; x < p.picWidth; x++ )
    {
      const int set = p.ctuSetIdx[ctuOf( p, x, yb )];
      if( set < 0 ) continue;
      const uint8_t c   = cls[( yb >> 2 ) * clsStride + ( x >> 2 )];
      const int classIdx = c & 31, tr = c >> 5;
      const int16_t* cf = coefTab + set * kSetStride + classIdx * kNumCoeff;
      const int16_t* cl = clipTab + set * kSetStride + classIdx * kNumCoeff;
      out[yb * outStride + x] = filterLumaSample( in + yb * inStride + x, inStride, p.bandY0 + yb, cf, cl, tr, p.vbH, p.vbPos, maxVal );
    }
  }
}

// copies the rectangles of CTUs with setIdx >= 0 from (src,srcStride) to (dst,dstStride); both point to (0,bandY0)
static void copyCtuAreas( const BandParams& p, const Pel* src, ptrdiff_t srcStride, Pel* dst, ptrdiff_t dstStride )
{
  const int rows = ( p.bandH + p.ctuSize - 1 ) / p.ctuSize;
  for( int r = 0; r < rows; r++ )
  {
    const int y0 = r * p.ctuSize, h = std::min( p.ctuSize, p.bandH - y0 );
    for( int c = 0; c < p.numCtuCols; c++ )
    {
      if( p.ctuSetIdx[r * p.numCtuCols + c] < 0 ) continue;
      const int x0 = c * p.ctuSize, w = std::min( p.ctuSize, p.picWidth - x0 );
      for( int y = y0; y < y0 + h; y++ )
      {
        memcpy( dst + y * dstStride + x0, src + y * srcStride + x0, w * sizeof( Pel ) );
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
// CPU simulation of the CUDA kernels: loops over the grid and over the threads of each block,
// executing the phases of AlfGpuTiles.h in the same order as the __syncthreads()-separated kernel
// ------------------------------------------------------------------------------------------------
void simClassifyKernel( ClsSmem& s, const Pel* d_in, const int16_t* d_ctuSet, uint8_t* d_cls, const KParams& k )
{
  const int gx = ( k.picWidth + CT_W - 1 ) / CT_W, gy = ( k.bandH + CT_H - 1 ) / CT_H;
  for( int by = 0; by < gy; by++ )
    for( int bx = 0; bx < gx; bx++ )
    {
      const int x0 = bx * CT_W, y0 = by * CT_H;
      for( int t = 0; t < CT_THREADS; t++ ) clsPhaseLoad( s, t, d_in, k, x0, y0 );
      for( int t = 0; t < CT_THREADS; t++ ) clsPhaseCells( s, t, k, y0 );
      for( int t = 0; t < CT_THREADS; t++ ) clsPhaseBlocks( s, t, d_ctuSet, d_cls, k, x0, y0 );
    }
}

void simFilterKernel( FltSmem& s, const Pel* d_in, const int16_t* d_ctuSet, const uint8_t* d_cls, const int16_t* d_coef,
                      const int16_t* d_clip, Pel* d_out, const KParams& k )
{
  const int gx = ( k.picWidth + FT_W - 1 ) / FT_W, gy = ( k.bandH + FT_H - 1 ) / FT_H;
  for( int by = 0; by < gy; by++ )
    for( int bx = 0; bx < gx; bx++ )
    {
      const int x0 = bx * FT_W, y0 = by * FT_H;
      for( int t = 0; t < FT_THREADS; t++ ) fltPhaseLoad( s, t, d_in, k, x0, y0 );
      for( int ty = 0; ty < FT_TY; ty++ )
        for( int tx = 0; tx < FT_TX; tx++ ) fltPhaseCompute( s, tx, ty, d_ctuSet, d_cls, d_coef, d_clip, d_out, k, x0, y0 );
    }
}

void simCovKernel( CovSmem& s, const CovKParams& k, int numUnits, const Pel* rec, const Pel* org, const uint8_t* cls,
                   const int* unitStart, const int16_t* ctuRect, const uint32_t* elem, float* out, uint8_t* used )
{
  const int groups = ( k.numElem + CV_THREADS - 1 ) / CV_THREADS;
  std::vector<float>    acc( ( size_t ) CV_THREADS * CV_MAXCLS );
  std::vector<uint32_t> mask( CV_THREADS );
  for( int unit = 0; unit < numUnits; unit++ )
    for( int g = 0; g < groups; g++ )
    {
      std::fill( acc.begin(), acc.end(), 0.f );
      std::fill( mask.begin(), mask.end(), 0u );
      for( int ctu = unitStart[unit]; ctu < unitStart[unit + 1]; ctu++ )
      {
        const int x0 = ctuRect[4 * ctu], y0 = ctuRect[4 * ctu + 1], bw = ctuRect[4 * ctu + 2] >> 2, bh = ctuRect[4 * ctu + 3] >> 2;
        for( int q0 = 0; q0 < bw * bh; q0 += CV_NB )
        {
          const int nb = std::min( CV_NB, bw * bh - q0 );
          for( int t = 0; t < CV_THREADS; t++ ) covPhaseLoad( s, t, k, rec, org, cls, x0, y0, bw, q0, nb );
          for( int t = 0; t < CV_THREADS; t++ )
          {
            const int e = g * CV_THREADS + t;
            if( e < k.numElem ) covPhaseAccum( s, elem[e], nb, &acc[( size_t ) t * CV_MAXCLS], mask[t] );
          }
        }
      }
      for( int t = 0; t < CV_THREADS; t++ )
      {
        const int e = g * CV_THREADS + t;
        if( e >= k.numElem ) continue;
        for( int c = 0; c < k.numClasses; c++ ) out[( ( size_t ) unit * k.numClasses + c ) * k.numElem + e] = acc[( size_t ) t * CV_MAXCLS + c];
      }
      if( g == 0 )
      {
        for( int c = 0; c < k.numClasses; c++ ) used[unit * k.numClasses + c] = ( mask[0] >> c ) & 1;
      }
    }
}

class HostEmuBackend : public Backend
{
  struct Worker
  {
    std::vector<Pel>     band;   // input band with halo ("device" input)
    std::vector<Pel>     out;    // filtered band ("device" output)
    std::vector<uint8_t> cls;    // classes ("device")
    std::unique_ptr<ClsSmem> clsSmem{ new ClsSmem };
    std::unique_ptr<FltSmem> fltSmem{ new FltSmem };
    std::unique_ptr<CovSmem> covSmem{ new CovSmem };
    std::vector<Pel>     plane[3], orgPlane[3];   // "device" chroma inputs (with halo) and original planes
  };
  std::vector<int16_t> m_coef, m_clip;   // kMaxSets * kSetStride
  std::vector<std::unique_ptr<Worker>> m_workers, m_slots;
  std::mutex m_mutex;

  using clk = std::chrono::steady_clock;
  static double ms( clk::time_point a, clk::time_point b ) { return std::chrono::duration<double, std::milli>( b - a ).count(); }

  Worker* get( std::vector<std::unique_ptr<Worker>>& v, int idx )
  {
    std::lock_guard<std::mutex> lock( m_mutex );
    if( idx >= ( int ) v.size() ) v.resize( idx + 1 );
    if( !v[idx] ) v[idx].reset( new Worker );
    return v[idx].get();
  }

  static KParams chunkParams( const BandParams& p, int cy, int chunkH, int inW )
  {
    KParams k;
    k.picWidth = p.picWidth;  k.bandY0 = p.bandY0 + cy;  k.bandH = std::min( chunkH, p.bandH - cy );  k.ctuSize = p.ctuSize;
    k.numCtuCols = p.numCtuCols;  k.bitDepth = p.bitDepth;  k.vbH = p.vbH;  k.vbPos = p.vbPos;
    k.inPitch = inW;  k.clsStride = p.picWidth >> 2;  k.outPitch = p.picWidth;
    return k;
  }

  // "H2D": the band with its halo into the worker's "device" buffer
  static void upload( Worker* w, const BandParams& p )
  {
    const int inW = p.picWidth + 2 * kHalo, inH = p.bandH + 2 * kHalo;
    w->band.resize( ( size_t ) inW * inH );
    for( int y = 0; y < inH; y++ )
    {
      memcpy( &w->band[( size_t ) y * inW], p.src + ( ptrdiff_t ) ( y - kHalo ) * p.srcStride - kHalo, inW * sizeof( Pel ) );
    }
  }

  void coefTables( const BandParams& p, std::vector<int16_t>& coef, std::vector<int16_t>& clip ) const
  {
    coef = m_coef; clip = m_clip;
    if( p.numAps > 0 )
    {
      memcpy( &coef[kNumFixedSets * kSetStride], p.apsCoef, p.numAps * kSetStride * sizeof( int16_t ) );
      memcpy( &clip[kNumFixedSets * kSetStride], p.apsClip, p.numAps * kSetStride * sizeof( int16_t ) );
    }
  }

  // kernels, chunk by chunk exactly as the CUDA backend splits the band
  static int runKernels( Worker* w, const BandParams& p, bool doClassify, bool doFilter, const int16_t* coef, const int16_t* clip,
                         double& clsMs, double& fltMs )
  {
    const int inW = p.picWidth + 2 * kHalo, clsStride = p.picWidth >> 2;
    w->cls.resize( ( size_t ) clsStride * ( ( p.bandH + 3 ) >> 2 ) );
    if( doFilter ) w->out.resize( ( size_t ) p.picWidth * p.bandH );
    const int chunkH = chunkCtuRows( p.bandH, p.ctuSize ) * p.ctuSize;
    int chunks = 0;
    for( int cy = 0; cy < p.bandH; cy += chunkH, chunks++ )
    {
      const KParams  k   = chunkParams( p, cy, chunkH, inW );
      const Pel*     in  = &w->band[( size_t ) cy * inW];                    // halo top of the chunk
      const int16_t* map = p.ctuSetIdx + ( cy / p.ctuSize ) * p.numCtuCols;
      uint8_t*       cl  = &w->cls[( size_t ) ( cy >> 2 ) * clsStride];
      const auto a = clk::now();
      if( doClassify ) simClassifyKernel( *w->clsSmem, in, map, cl, k );
      const auto b = clk::now();
      if( doFilter ) simFilterKernel( *w->fltSmem, in, map, cl, coef, clip, &w->out[( size_t ) cy * p.picWidth], k );
      const auto c = clk::now();
      clsMs += ms( a, b ); fltMs += ms( b, c );
    }
    return chunks;
  }

public:
  const char* name() const override { return "host-emu"; }

  bool init( const int16_t* fixedCoef, const int16_t* fixedClip, int, int, int numWorkers ) override
  {
    m_coef.assign( kMaxSets * kSetStride, 0 );
    m_clip.assign( kMaxSets * kSetStride, 0 );
    memcpy( m_coef.data(), fixedCoef, kNumFixedSets * kSetStride * sizeof( int16_t ) );
    memcpy( m_clip.data(), fixedClip, kNumFixedSets * kSetStride * sizeof( int16_t ) );
    m_workers.resize( std::max( 1, numWorkers ) );
    return true;
  }

  void filterLumaBand( const BandParams& p, int mode, int worker, Timing& t ) override
  {
    const auto t0 = clk::now();
    Worker* w = get( m_workers, worker );
    std::vector<int16_t> coef, clip;
    coefTables( p, coef, clip );

    const auto t1 = clk::now();
    upload( w, p );
    if( mode == MODE_FILTER )
    {
      w->cls.resize( ( size_t ) ( p.picWidth >> 2 ) * ( ( p.bandH + 3 ) >> 2 ) );
      memcpy( w->cls.data(), p.cpuClass, w->cls.size() );
    }
    const auto t2 = clk::now();
    double clsMs = 0, fltMs = 0;
    const int chunks = runKernels( w, p, mode != MODE_FILTER, true, coef.data(), clip.data(), clsMs, fltMs );
    const auto t4 = clk::now();
    copyCtuAreas( p, w->out.data(), p.picWidth, p.dst, p.dstStride );   // "D2H" + unpack
    if( p.outClass ) memcpy( p.outClass, w->cls.data(), w->cls.size() );
    const auto t5 = clk::now();

    t.h2dMs        += ms( t1, t2 );
    t.classifyMs   += clsMs;
    t.filterMs     += fltMs;
    t.hostUnpackMs += ms( t4, t5 );
    t.gpuSpanMs    += ms( t1, t4 );
    t.totalMs      += ms( t0, t5 );
    t.chunks       += chunks;
    t.bytesH2D     += ( uint64_t ) ( p.picWidth + 2 * kHalo ) * ( p.bandH + 2 * kHalo ) * sizeof( Pel );
    t.bytesD2H     += ( uint64_t ) p.picWidth * p.bandH * sizeof( Pel );
  }

  void classifyFrame( const BandParams& p, int slot, Timing& t ) override
  {
    const auto t0 = clk::now();
    Worker* w = get( m_slots, slot );
    upload( w, p );
    const auto t1 = clk::now();
    double clsMs = 0, fltMs = 0;
    const int chunks = runKernels( w, p, true, false, nullptr, nullptr, clsMs, fltMs );
    const auto t2 = clk::now();
    memcpy( p.outClass, w->cls.data(), w->cls.size() );   // "D2H" of the classes
    const auto t3 = clk::now();

    t.h2dMs      += ms( t0, t1 );
    t.classifyMs += clsMs;
    t.d2hMs      += ms( t2, t3 );
    t.gpuSpanMs  += ms( t0, t3 );
    t.totalMs    += ms( t0, t3 );
    t.chunks     += chunks;
    t.bytesH2D   += ( uint64_t ) ( p.picWidth + 2 * kHalo ) * ( p.bandH + 2 * kHalo ) * sizeof( Pel );
    t.bytesD2H   += w->cls.size();
  }

  void covarianceFrame( const CovJob* jobs, int numJobs, int slot, Timing& t ) override
  {
    const auto t0 = clk::now();
    Worker* w = get( m_slots, slot );
    double upMs = 0, kMs = 0;
    for( int j = 0; j < numJobs; j++ )
    {
      const CovJob& jb = jobs[j];
      CovKParams k = jb.k;
      const Pel* rec;
      const uint8_t* cls = nullptr;
      const auto a = clk::now();
      // "H2D": original plane (and chroma reconstruction with halo)
      std::vector<Pel>& og = w->orgPlane[jb.comp];
      og.resize( ( size_t ) jb.width * jb.height );
      for( int y = 0; y < jb.height; y++ ) memcpy( &og[( size_t ) y * jb.width], jb.org + y * jb.orgStride, jb.width * sizeof( Pel ) );
      k.orgPitch = jb.width;
      if( jb.comp == 0 )
      {
        k.recPitch  = jb.width + 2 * kHalo;
        k.clsStride = jb.width >> 2;
        rec = &w->band[( size_t ) kHalo * k.recPitch + kHalo];      // resident from classifyFrame
        cls = w->cls.data();
      }
      else
      {
        k.recPitch = jb.width + 2 * kHalo;
        std::vector<Pel>& pl = w->plane[jb.comp];
        pl.resize( ( size_t ) k.recPitch * ( jb.height + 2 * kHalo ) );
        for( int y = -kHalo; y < jb.height + kHalo; y++ )
        {
          memcpy( &pl[( size_t ) ( y + kHalo ) * k.recPitch], jb.rec + y * jb.recStride - kHalo, k.recPitch * sizeof( Pel ) );
        }
        rec = &pl[( size_t ) kHalo * k.recPitch + kHalo];
        t.bytesH2D += pl.size() * sizeof( Pel );
      }
      t.bytesH2D += og.size() * sizeof( Pel );
      const auto b = clk::now();
      simCovKernel( *w->covSmem, k, jb.numUnits, rec, og.data(), cls, jb.unitCtuStart, jb.ctuRect, jb.elem, jb.out, jb.used );
      const auto c = clk::now();
      upMs += ms( a, b ); kMs += ms( b, c );
      t.bytesD2H += ( uint64_t ) jb.numUnits * k.numClasses * ( k.numElem * sizeof( float ) + 1 );
    }
    const auto t1 = clk::now();
    t.h2dMs     += upMs;
    t.covMs     += kMs;
    t.gpuSpanMs += ms( t0, t1 );
    t.totalMs   += ms( t0, t1 );
  }

  void filterFrameResident( const BandParams& p, int slot, Timing& t ) override
  {
    const auto t0 = clk::now();
    Worker* w = get( m_slots, slot );
    std::vector<int16_t> coef, clip;
    coefTables( p, coef, clip );
    const auto t1 = clk::now();
    double clsMs = 0, fltMs = 0;
    const int chunks = runKernels( w, p, false, true, coef.data(), clip.data(), clsMs, fltMs );
    const auto t2 = clk::now();
    copyCtuAreas( p, w->out.data(), p.picWidth, p.dst, p.dstStride );
    const auto t3 = clk::now();

    t.hostPackMs   += ms( t0, t1 );
    t.filterMs     += fltMs;
    t.hostUnpackMs += ms( t2, t3 );
    t.gpuSpanMs    += ms( t1, t2 );
    t.totalMs      += ms( t0, t3 );
    t.chunks       += chunks;
    t.bytesH2D     += ( uint64_t ) ( ( p.bandH + p.ctuSize - 1 ) / p.ctuSize ) * p.numCtuCols * sizeof( int16_t )
                    + ( uint64_t ) 2 * p.numAps * kSetStride * sizeof( int16_t );
    t.bytesD2H     += ( uint64_t ) p.picWidth * p.bandH * sizeof( Pel );
  }
};

#if VVENC_CUDA_ALF
Backend* createCudaBackend();   // AlfGpuCuda.cu, returns nullptr if no usable device
#endif

Backend* createBackend( const int16_t* fixedCoef, const int16_t* fixedClip, int maxPicWidth, int maxBandH, int numWorkers )
{
  const char* emu = getenv( "VVENC_ALF_GPU_EMU" );
#if VVENC_CUDA_ALF
  if( !( emu && atoi( emu ) ) )
  {
    Backend* b = createCudaBackend();
    if( b && b->init( fixedCoef, fixedClip, maxPicWidth, maxBandH, numWorkers ) ) return b;
    delete b;
    fprintf( stderr, "[AlfGpu] no usable CUDA device, falling back to host emulation\n" );
  }
#else
  ( void ) emu;
#endif
  Backend* b = new HostEmuBackend;
  b->init( fixedCoef, fixedClip, maxPicWidth, maxBandH, numWorkers );
  return b;
}

} // namespace alfgpu
