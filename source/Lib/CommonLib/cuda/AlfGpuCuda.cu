/* -----------------------------------------------------------------------------
AlfGpuCuda.cu - CUDA backend for the VVenC ALF luma reconstruction

Per worker (encoder thread, or picture slot of the frame-resident path) it owns: CUDA
streams, pinned staging buffers, device buffers and a device coefficient table (16
fixed sets uploaded once + APS sets of the current picture uploaded per call).

Frame-resident path (--AlfGpu 1, default): classifyFrame() at the start of the ALF
uploads the picture once and computes gradients / classes / transpose indices; the
encoder uses the classes for its statistics. filterFrameResident() after the filter
decisions reuses the picture and the classes already on the device: only the CTU map
and the APS sets go up, only the filtered picture comes back.

A band (one CTU row, or the whole frame in frame mode) is split into chunks of CTU
rows that run as a software pipeline over several streams:
    host pack(c) -> H2D(c)        (the CPU packs chunk c+1 while chunk c is copied)
    kernels(c) wait for H2D(c-1, c, c+1) (3-sample halo from the neighbour chunks)
    D2H(c) -> host unpack(c)      (unpack of chunk c overlaps the GPU work of c+1..)
so transfers, kernels and host copies of different chunks overlap.
env: VVENC_ALF_GPU_STREAMS (default 2), VVENC_ALF_GPU_CHUNK_ROWS (CTU rows per chunk).
Every stage of every chunk is timed with CUDA events; GPU energy is read through NVML.
------------------------------------------------------------------------------- */

#include "../AlfGpu.h"
#include "AlfGpuKernels.cuh"

#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace alfgpu
{

#define ALF_CUDA_CHECK( x )                                                                       \
  do {                                                                                            \
    cudaError_t e_ = ( x );                                                                       \
    if( e_ != cudaSuccess ) {                                                                     \
      fprintf( stderr, "[AlfGpu] CUDA error %s at %s:%d\n", cudaGetErrorString( e_ ), __FILE__, __LINE__ ); \
      abort();                                                                                    \
    }                                                                                             \
  } while( 0 )

class CudaBackend : public Backend
{
  static constexpr int kMaxStreams = 4;

  struct Worker
  {
    cudaStream_t stream[kMaxStreams] = {};
    cudaEvent_t  evMeta0 = nullptr, evMeta1 = nullptr;
    std::vector<cudaEvent_t> ev;                  // 6 per chunk: H2D start/end, kernel start, after classify, after filter, after D2H
    size_t capIn = 0, capOut = 0, capCls = 0, capMap = 0;
    Pel     *hIn = nullptr, *hOut = nullptr, *dIn = nullptr, *dOut = nullptr;
    uint8_t *hCls = nullptr, *dCls = nullptr;
    int16_t *hMap = nullptr, *dMap = nullptr;
    int16_t *hAps = nullptr;                      // pinned: [coef | clip] of APS sets
    int16_t *dCoef = nullptr, *dClip = nullptr;   // kMaxSets * kSetStride each
    int residentW = 0, residentH = 0;             // picture currently resident in dIn / dCls (frame-resident path)
    // covariance statistics, per colour component
    struct Buf { void* h = nullptr; void* d = nullptr; size_t cap = 0; };
    Buf cvOrg[3], cvRec[3], cvUnit[3], cvRect[3], cvElem[3], cvOut[3], cvUsed[3];
    cudaEvent_t cvEv[3][4] = {};

    void ensureEvents( size_t n )
    {
      while( ev.size() < n )
      {
        cudaEvent_t e;
        ALF_CUDA_CHECK( cudaEventCreate( &e ) );
        ev.push_back( e );
      }
    }

    static void growBuf( Buf& b, size_t bytes )
    {
      if( bytes <= b.cap ) return;
      if( b.h ) cudaFreeHost( b.h );
      if( b.d ) cudaFree( b.d );
      ALF_CUDA_CHECK( cudaMallocHost( &b.h, bytes ) );
      ALF_CUDA_CHECK( cudaMalloc( &b.d, bytes ) );
      b.cap = bytes;
    }

    ~Worker()
    {
      if( !stream[0] ) return;
      for( int c = 0; c < 3; c++ )
      {
        for( Buf* b : { &cvOrg[c], &cvRec[c], &cvUnit[c], &cvRect[c], &cvElem[c], &cvOut[c], &cvUsed[c] } )
        {
          if( b->h ) cudaFreeHost( b->h );
          if( b->d ) cudaFree( b->d );
        }
        for( auto e : cvEv[c] ) if( e ) cudaEventDestroy( e );
      }
      for( auto s : stream ) if( s ) cudaStreamSynchronize( s );
      for( auto e : ev ) cudaEventDestroy( e );
      cudaEventDestroy( evMeta0 ); cudaEventDestroy( evMeta1 );
      cudaFreeHost( hIn ); cudaFreeHost( hOut ); cudaFreeHost( hCls ); cudaFreeHost( hMap ); cudaFreeHost( hAps );
      cudaFree( dIn ); cudaFree( dOut ); cudaFree( dCls ); cudaFree( dMap ); cudaFree( dCoef ); cudaFree( dClip );
      for( auto s : stream ) if( s ) cudaStreamDestroy( s );
    }
  };

  int m_device = 0;
  int m_numStreams = 2;
  int m_slotBase   = 1;                           // workers [0, m_slotBase) = encoder threads, then one per slot
  std::vector<int16_t> m_fixedCoef, m_fixedClip;
  std::vector<std::unique_ptr<Worker>> m_workers;
  std::mutex m_mutex;
  bool m_nvmlOk = false;
  nvmlDevice_t m_nvmlDev{};

  template<typename T> static void growHost( T*& p, size_t& cap, size_t n )
  {
    if( n <= cap ) return;
    if( p ) ALF_CUDA_CHECK( cudaFreeHost( p ) );
    ALF_CUDA_CHECK( cudaMallocHost( ( void** ) &p, n * sizeof( T ) ) );
    cap = n;
  }
  template<typename T> static void growDev( T*& p, size_t n, size_t oldCap )
  {
    if( n <= oldCap && p ) return;
    if( p ) ALF_CUDA_CHECK( cudaFree( p ) );
    ALF_CUDA_CHECK( cudaMalloc( ( void** ) &p, n * sizeof( T ) ) );
  }

  Worker* getWorker( int idx )
  {
    std::lock_guard<std::mutex> lock( m_mutex );
    if( idx >= ( int ) m_workers.size() ) m_workers.resize( idx + 1 );
    if( !m_workers[idx] )
    {
      ALF_CUDA_CHECK( cudaSetDevice( m_device ) );
      Worker* w = new Worker;
      for( int i = 0; i < m_numStreams; i++ ) ALF_CUDA_CHECK( cudaStreamCreateWithFlags( &w->stream[i], cudaStreamNonBlocking ) );
      ALF_CUDA_CHECK( cudaEventCreate( &w->evMeta0 ) );
      ALF_CUDA_CHECK( cudaEventCreate( &w->evMeta1 ) );
      const size_t tab = ( size_t ) kMaxSets * kSetStride;
      ALF_CUDA_CHECK( cudaMalloc( ( void** ) &w->dCoef, tab * sizeof( int16_t ) ) );
      ALF_CUDA_CHECK( cudaMalloc( ( void** ) &w->dClip, tab * sizeof( int16_t ) ) );
      ALF_CUDA_CHECK( cudaMemset( w->dCoef, 0, tab * sizeof( int16_t ) ) );
      ALF_CUDA_CHECK( cudaMemset( w->dClip, 0, tab * sizeof( int16_t ) ) );
      ALF_CUDA_CHECK( cudaMemcpy( w->dCoef, m_fixedCoef.data(), m_fixedCoef.size() * sizeof( int16_t ), cudaMemcpyHostToDevice ) );
      ALF_CUDA_CHECK( cudaMemcpy( w->dClip, m_fixedClip.data(), m_fixedClip.size() * sizeof( int16_t ), cudaMemcpyHostToDevice ) );
      ALF_CUDA_CHECK( cudaMallocHost( ( void** ) &w->hAps, 2 * kMaxAps * kSetStride * sizeof( int16_t ) ) );
      m_workers[idx].reset( w );
    }
    return m_workers[idx].get();
  }

public:
  ~CudaBackend() override
  {
    m_workers.clear();
    if( m_nvmlOk ) nvmlShutdown();
  }

  const char* name() const override { return "cuda"; }

  bool init( const int16_t* fixedCoef, const int16_t* fixedClip, int, int, int numWorkers ) override
  {
    int n = 0;
    if( cudaGetDeviceCount( &n ) != cudaSuccess || n == 0 ) return false;
    if( const char* d = getenv( "VVENC_ALF_GPU_DEVICE" ) ) m_device = std::max( 0, std::min( n - 1, atoi( d ) ) );
    if( const char* s = getenv( "VVENC_ALF_GPU_STREAMS" ) ) m_numStreams = std::max( 1, std::min( kMaxStreams, atoi( s ) ) );
    ALF_CUDA_CHECK( cudaSetDevice( m_device ) );
    cudaDeviceProp prop;
    ALF_CUDA_CHECK( cudaGetDeviceProperties( &prop, m_device ) );
    fprintf( stderr, "[AlfGpu] CUDA device %d: %s (sm_%d%d, %d SMs), %d stream(s)\n", m_device, prop.name, prop.major, prop.minor,
             prop.multiProcessorCount, m_numStreams );

    m_fixedCoef.assign( fixedCoef, fixedCoef + kNumFixedSets * kSetStride );
    m_fixedClip.assign( fixedClip, fixedClip + kNumFixedSets * kSetStride );
    m_workers.resize( std::max( 1, numWorkers ) );
    m_slotBase = std::max( 1, numWorkers );

    if( nvmlInit_v2() == NVML_SUCCESS )
    {
      char pci[32];
      if( cudaDeviceGetPCIBusId( pci, sizeof( pci ), m_device ) == cudaSuccess &&
          nvmlDeviceGetHandleByPciBusId_v2( pci, &m_nvmlDev ) == NVML_SUCCESS )
      {
        m_nvmlOk = true;
      }
    }
    // load the kernels now (lazy module loading would otherwise charge the first frame)
    cudaFuncAttributes attr;
    cudaFuncGetAttributes( &attr, kAlfClassify );
    cudaFuncGetAttributes( &attr, kAlfFilterLuma );
    cudaFuncGetAttributes( &attr, kAlfCovariance );
    return true;
  }

  bool readEnergy_mJ( unsigned long long& e ) override
  {
    return m_nvmlOk && nvmlDeviceGetTotalEnergyConsumption( m_nvmlDev, &e ) == NVML_SUCCESS;
  }
  bool readPower_mW( unsigned& p ) override
  {
    return m_nvmlOk && nvmlDeviceGetPowerUsage( m_nvmlDev, &p ) == NVML_SUCCESS;
  }

  // what one call of the generic pipeline does
  enum Step : unsigned
  {
    UP_IN    = 1,    // pack + H2D of the input band (else: the slot's resident input is used)
    UP_CLS   = 2,    // H2D of p.cpuClass
    CLASSIFY = 4,
    FILTER   = 8,    // filter kernel + D2H + unpack of the filtered CTUs
    DN_CLS   = 16,   // D2H of the classes into p.outClass
  };

  void filterLumaBand( const BandParams& p, int mode, int worker, Timing& t ) override
  {
    unsigned steps = UP_IN | FILTER;
    if( mode == MODE_FILTER ) steps |= UP_CLS; else steps |= CLASSIFY;
    if( p.outClass && mode != MODE_FILTER ) steps |= DN_CLS;
    run( p, getWorker( worker ), steps, t );
  }

  void classifyFrame( const BandParams& p, int slot, Timing& t ) override
  {
    run( p, getWorker( m_slotBase + slot ), UP_IN | CLASSIFY | DN_CLS, t );
  }

  void filterFrameResident( const BandParams& p, int slot, Timing& t ) override
  {
    run( p, getWorker( m_slotBase + slot ), FILTER, t );
  }

  void covarianceFrame( const CovJob* jobs, int numJobs, int slot, Timing& t ) override
  {
    using clk = std::chrono::steady_clock;
    auto ms   = []( clk::time_point a, clk::time_point b ) { return std::chrono::duration<double, std::milli>( b - a ).count(); };
    const auto t0 = clk::now();
    Worker* w = getWorker( m_slotBase + slot );
    ALF_CUDA_CHECK( cudaSetDevice( m_device ) );
    double packMs = 0, unpackMs = 0;

    // one stream per colour component: pack + H2D, kernel, D2H; the components overlap
    for( int j = 0; j < numJobs; j++ )
    {
      const CovJob& jb = jobs[j];
      const int c = jb.comp;
      cudaStream_t st = w->stream[j % m_numStreams];
      for( auto& e : w->cvEv[c] ) if( !e ) ALF_CUDA_CHECK( cudaEventCreate( &e ) );
      CovKParams k = jb.k;
      const size_t orgN  = ( size_t ) jb.width * jb.height;
      const int    nCtu  = jb.unitCtuStart[jb.numUnits];
      const size_t outN  = ( size_t ) jb.numUnits * k.numClasses * k.numElem;
      const size_t usedN = ( size_t ) jb.numUnits * k.numClasses;
      Worker::growBuf( w->cvOrg[c],  orgN * sizeof( Pel ) );
      Worker::growBuf( w->cvUnit[c], ( jb.numUnits + 1 ) * sizeof( int ) );
      Worker::growBuf( w->cvRect[c], ( size_t ) nCtu * 4 * sizeof( int16_t ) );
      Worker::growBuf( w->cvElem[c], k.numElem * sizeof( uint32_t ) );
      Worker::growBuf( w->cvOut[c],  outN * sizeof( float ) );
      Worker::growBuf( w->cvUsed[c], usedN );

      const auto a = clk::now();
      Pel* hOrg = ( Pel* ) w->cvOrg[c].h;
      for( int y = 0; y < jb.height; y++ ) memcpy( hOrg + ( size_t ) y * jb.width, jb.org + y * jb.orgStride, jb.width * sizeof( Pel ) );
      memcpy( w->cvUnit[c].h, jb.unitCtuStart, ( jb.numUnits + 1 ) * sizeof( int ) );
      memcpy( w->cvRect[c].h, jb.ctuRect, ( size_t ) nCtu * 4 * sizeof( int16_t ) );
      memcpy( w->cvElem[c].h, jb.elem, k.numElem * sizeof( uint32_t ) );
      k.orgPitch = jb.width;
      k.recPitch = jb.width + 2 * kHalo;
      const Pel* dRec;
      const uint8_t* dCls = nullptr;
      size_t recBytes = 0;
      if( c == 0 )
      {
        if( !w->dIn || w->residentW != jb.width || w->residentH != jb.height )
        {
          fprintf( stderr, "[AlfGpu] covarianceFrame (luma) without a matching classifyFrame\n" );
          abort();
        }
        k.clsStride = jb.width >> 2;
        dRec = w->dIn + ( size_t ) kHalo * k.recPitch + kHalo;
        dCls = w->dCls;
      }
      else
      {
        recBytes = ( size_t ) k.recPitch * ( jb.height + 2 * kHalo ) * sizeof( Pel );
        Worker::growBuf( w->cvRec[c], recBytes );
        Pel* hRec = ( Pel* ) w->cvRec[c].h;
        for( int y = -kHalo; y < jb.height + kHalo; y++ )
        {
          memcpy( hRec + ( size_t ) ( y + kHalo ) * k.recPitch, jb.rec + y * jb.recStride - kHalo, k.recPitch * sizeof( Pel ) );
        }
        dRec = ( const Pel* ) w->cvRec[c].d + ( size_t ) kHalo * k.recPitch + kHalo;
      }
      packMs += ms( a, clk::now() );

      ALF_CUDA_CHECK( cudaEventRecord( w->cvEv[c][0], st ) );
      ALF_CUDA_CHECK( cudaMemcpyAsync( w->cvOrg[c].d, w->cvOrg[c].h, orgN * sizeof( Pel ), cudaMemcpyHostToDevice, st ) );
      if( recBytes ) ALF_CUDA_CHECK( cudaMemcpyAsync( w->cvRec[c].d, w->cvRec[c].h, recBytes, cudaMemcpyHostToDevice, st ) );
      ALF_CUDA_CHECK( cudaMemcpyAsync( w->cvUnit[c].d, w->cvUnit[c].h, ( jb.numUnits + 1 ) * sizeof( int ), cudaMemcpyHostToDevice, st ) );
      ALF_CUDA_CHECK( cudaMemcpyAsync( w->cvRect[c].d, w->cvRect[c].h, ( size_t ) nCtu * 4 * sizeof( int16_t ), cudaMemcpyHostToDevice, st ) );
      ALF_CUDA_CHECK( cudaMemcpyAsync( w->cvElem[c].d, w->cvElem[c].h, k.numElem * sizeof( uint32_t ), cudaMemcpyHostToDevice, st ) );
      ALF_CUDA_CHECK( cudaEventRecord( w->cvEv[c][1], st ) );
      launchAlfCovariance( k, jb.numUnits, dRec, ( const Pel* ) w->cvOrg[c].d, dCls, ( const int* ) w->cvUnit[c].d,
                           ( const int16_t* ) w->cvRect[c].d, ( const uint32_t* ) w->cvElem[c].d, ( float* ) w->cvOut[c].d,
                           ( uint8_t* ) w->cvUsed[c].d, st );
      ALF_CUDA_CHECK( cudaGetLastError() );
      ALF_CUDA_CHECK( cudaEventRecord( w->cvEv[c][2], st ) );
      ALF_CUDA_CHECK( cudaMemcpyAsync( w->cvOut[c].h, w->cvOut[c].d, outN * sizeof( float ), cudaMemcpyDeviceToHost, st ) );
      ALF_CUDA_CHECK( cudaMemcpyAsync( w->cvUsed[c].h, w->cvUsed[c].d, usedN, cudaMemcpyDeviceToHost, st ) );
      ALF_CUDA_CHECK( cudaEventRecord( w->cvEv[c][3], st ) );
      t.bytesH2D += orgN * sizeof( Pel ) + recBytes + ( jb.numUnits + 1 ) * sizeof( int ) + ( size_t ) nCtu * 8 + k.numElem * 4;
      t.bytesD2H += outN * sizeof( float ) + usedN;
    }

    // wait per component, unpack, collect the event times
    double sH2D = 0, sCov = 0, sD2H = 0, span = 0;
    for( int j = 0; j < numJobs; j++ )
    {
      const CovJob& jb = jobs[j];
      const int c = jb.comp;
      cudaStream_t st = w->stream[j % m_numStreams];
      ALF_CUDA_CHECK( cudaStreamSynchronize( st ) );
      const auto a = clk::now();
      const size_t outN = ( size_t ) jb.numUnits * jb.k.numClasses * jb.k.numElem, usedN = ( size_t ) jb.numUnits * jb.k.numClasses;
      memcpy( jb.out, w->cvOut[c].h, outN * sizeof( float ) );
      memcpy( jb.used, w->cvUsed[c].h, usedN );
      unpackMs += ms( a, clk::now() );
      float x, y, z, sp;
      cudaEventElapsedTime( &x, w->cvEv[c][0], w->cvEv[c][1] );
      cudaEventElapsedTime( &y, w->cvEv[c][1], w->cvEv[c][2] );
      cudaEventElapsedTime( &z, w->cvEv[c][2], w->cvEv[c][3] );
      cudaEventElapsedTime( &sp, w->cvEv[jobs[0].comp][0], w->cvEv[c][3] );
      sH2D += x; sCov += y; sD2H += z; span = std::max<double>( span, sp );
    }
    const auto t1 = clk::now();
    t.hostPackMs   += packMs;
    t.h2dMs        += sH2D;
    t.covMs        += sCov;
    t.d2hMs        += sD2H;
    t.hostUnpackMs += unpackMs;
    t.gpuSpanMs    += span;
    t.totalMs      += ms( t0, t1 );
  }

private:
  void run( const BandParams& p, Worker* w, const unsigned steps, Timing& t )
  {
    using clk = std::chrono::steady_clock;
    auto ms   = []( clk::time_point a, clk::time_point b ) { return std::chrono::duration<double, std::milli>( b - a ).count(); };
    const auto t0 = clk::now();
    ALF_CUDA_CHECK( cudaSetDevice( m_device ) );

    const int    inPitch   = p.picWidth + 2 * kHalo;
    const size_t inN       = ( size_t ) inPitch * ( p.bandH + 2 * kHalo );
    const size_t outN      = ( size_t ) p.picWidth * p.bandH;
    const int    clsStride = p.picWidth >> 2;
    const size_t clsN      = ( size_t ) clsStride * ( ( p.bandH + 3 ) >> 2 );
    const int    ctuRows   = ( p.bandH + p.ctuSize - 1 ) / p.ctuSize;
    const size_t mapN      = ( size_t ) ctuRows * p.numCtuCols;
    const int    chunkH    = chunkCtuRows( p.bandH, p.ctuSize ) * p.ctuSize;
    const int    nChunks   = ( p.bandH + chunkH - 1 ) / chunkH;
    const bool   upIn      = steps & UP_IN, filter = steps & FILTER, classify = steps & CLASSIFY;

    if( upIn )
    {
      size_t c = w->capIn; growHost( w->hIn, w->capIn, inN ); growDev( w->dIn, inN, c );
      w->residentH = p.bandH; w->residentW = p.picWidth;
    }
    else
    {
      if( !w->dIn || w->residentH != p.bandH || w->residentW != p.picWidth )
      {
        fprintf( stderr, "[AlfGpu] filterFrameResident without a matching classifyFrame\n" );
        abort();
      }
    }
    if( filter ) { size_t c = w->capOut; growHost( w->hOut, w->capOut, outN ); growDev( w->dOut, outN, c ); }
    if( upIn || ( steps & UP_CLS ) ) { size_t c = w->capCls; growHost( w->hCls, w->capCls, clsN ); growDev( w->dCls, clsN, c ); }
    { size_t c = w->capMap; growHost( w->hMap, w->capMap, mapN ); growDev( w->dMap, mapN, c ); }
    w->ensureEvents( ( size_t ) nChunks * 6 );
    auto EV = [&]( int c, int i ) { return w->ev[c * 6 + i]; };
    auto ST = [&]( int c ) { return w->stream[c % m_numStreams]; };

    double packMs = 0, unpackMs = 0;
    uint64_t h2d = 0, d2h = 0;

    // ---- per-call metadata (CTU map, APS sets, optional CPU classes) on stream 0; all chunks wait for it
    {
      const auto a = clk::now();
      memcpy( w->hMap, p.ctuSetIdx, mapN * sizeof( int16_t ) );
      const bool aps = filter && p.numAps > 0;
      if( aps )
      {
        memcpy( w->hAps, p.apsCoef, p.numAps * kSetStride * sizeof( int16_t ) );
        memcpy( w->hAps + kMaxAps * kSetStride, p.apsClip, p.numAps * kSetStride * sizeof( int16_t ) );
      }
      if( steps & UP_CLS ) memcpy( w->hCls, p.cpuClass, clsN );
      packMs += ms( a, clk::now() );

      cudaStream_t s = w->stream[0];
      ALF_CUDA_CHECK( cudaEventRecord( w->evMeta0, s ) );
      ALF_CUDA_CHECK( cudaMemcpyAsync( w->dMap, w->hMap, mapN * sizeof( int16_t ), cudaMemcpyHostToDevice, s ) );
      h2d += mapN * sizeof( int16_t );
      if( aps )
      {
        const size_t n = ( size_t ) p.numAps * kSetStride;
        ALF_CUDA_CHECK( cudaMemcpyAsync( w->dCoef + kNumFixedSets * kSetStride, w->hAps, n * sizeof( int16_t ), cudaMemcpyHostToDevice, s ) );
        ALF_CUDA_CHECK( cudaMemcpyAsync( w->dClip + kNumFixedSets * kSetStride, w->hAps + kMaxAps * kSetStride, n * sizeof( int16_t ), cudaMemcpyHostToDevice, s ) );
        h2d += 2 * n * sizeof( int16_t );
      }
      if( steps & UP_CLS )
      {
        ALF_CUDA_CHECK( cudaMemcpyAsync( w->dCls, w->hCls, clsN, cudaMemcpyHostToDevice, s ) );
        h2d += clsN;
      }
      ALF_CUDA_CHECK( cudaEventRecord( w->evMeta1, s ) );
    }

    // ---- pack + H2D per chunk (chunk c owns input rows [y0, y0+h); the first/last chunk also carry the outer halo)
    for( int c = 0; c < nChunks; c++ )
    {
      ALF_CUDA_CHECK( cudaEventRecord( EV( c, 0 ), ST( c ) ) );
      if( upIn )
      {
        const int y0   = c * chunkH, h = std::min( chunkH, p.bandH - y0 );
        const int rBeg = ( c == 0 ) ? -kHalo : y0;                       // band rows
        const int rEnd = ( c == nChunks - 1 ) ? p.bandH + kHalo : y0 + h;
        const auto a = clk::now();
        for( int y = rBeg; y < rEnd; y++ )
        {
          memcpy( w->hIn + ( size_t ) ( y + kHalo ) * inPitch, p.src + ( ptrdiff_t ) y * p.srcStride - kHalo, inPitch * sizeof( Pel ) );
        }
        packMs += ms( a, clk::now() );
        const size_t off = ( size_t ) ( rBeg + kHalo ) * inPitch, n = ( size_t ) ( rEnd - rBeg ) * inPitch;
        ALF_CUDA_CHECK( cudaMemcpyAsync( w->dIn + off, w->hIn + off, n * sizeof( Pel ), cudaMemcpyHostToDevice, ST( c ) ) );
        h2d += n * sizeof( Pel );
      }
      ALF_CUDA_CHECK( cudaEventRecord( EV( c, 1 ), ST( c ) ) );
    }

    // ---- kernels + D2H per chunk
    for( int c = 0; c < nChunks; c++ )
    {
      const int y0 = c * chunkH, h = std::min( chunkH, p.bandH - y0 );
      cudaStream_t s = ST( c );
      ALF_CUDA_CHECK( cudaStreamWaitEvent( s, w->evMeta1, 0 ) );
      if( upIn && c > 0 )           ALF_CUDA_CHECK( cudaStreamWaitEvent( s, EV( c - 1, 1 ), 0 ) );   // top halo
      if( upIn && c < nChunks - 1 ) ALF_CUDA_CHECK( cudaStreamWaitEvent( s, EV( c + 1, 1 ), 0 ) );   // bottom halo

      KParams k;
      k.picWidth = p.picWidth;  k.bandY0 = p.bandY0 + y0;  k.bandH = h;  k.ctuSize = p.ctuSize;
      k.numCtuCols = p.numCtuCols;  k.bitDepth = p.bitDepth;  k.vbH = p.vbH;  k.vbPos = p.vbPos;
      k.inPitch = inPitch;  k.clsStride = clsStride;  k.outPitch = p.picWidth;
      const Pel*     in  = w->dIn + ( size_t ) y0 * inPitch;              // halo top of the chunk
      const int16_t* map = w->dMap + ( y0 / p.ctuSize ) * p.numCtuCols;
      uint8_t*       cl  = w->dCls + ( size_t ) ( y0 >> 2 ) * clsStride;
      Pel*           out = filter ? w->dOut + ( size_t ) y0 * p.picWidth : nullptr;

      ALF_CUDA_CHECK( cudaEventRecord( EV( c, 2 ), s ) );
      if( classify ) launchAlfClassify( in, map, cl, k, s );
      ALF_CUDA_CHECK( cudaGetLastError() );
      ALF_CUDA_CHECK( cudaEventRecord( EV( c, 3 ), s ) );
      if( filter ) launchAlfFilterLuma( in, map, cl, w->dCoef, w->dClip, out, k, s );
      ALF_CUDA_CHECK( cudaGetLastError() );
      ALF_CUDA_CHECK( cudaEventRecord( EV( c, 4 ), s ) );
      if( filter )
      {
        ALF_CUDA_CHECK( cudaMemcpyAsync( w->hOut + ( size_t ) y0 * p.picWidth, out, ( size_t ) h * p.picWidth * sizeof( Pel ), cudaMemcpyDeviceToHost, s ) );
        d2h += ( size_t ) h * p.picWidth * sizeof( Pel );
      }
      if( steps & DN_CLS )
      {
        const size_t n = ( size_t ) clsStride * ( ( h + 3 ) >> 2 );
        ALF_CUDA_CHECK( cudaMemcpyAsync( w->hCls + ( size_t ) ( y0 >> 2 ) * clsStride, cl, n, cudaMemcpyDeviceToHost, s ) );
        d2h += n;
      }
      ALF_CUDA_CHECK( cudaEventRecord( EV( c, 5 ), s ) );
    }

    // ---- unpack chunk by chunk as they complete (overlaps with the GPU work of later chunks)
    for( int c = 0; c < nChunks; c++ )
    {
      ALF_CUDA_CHECK( cudaEventSynchronize( EV( c, 5 ) ) );
      const auto a = clk::now();
      const int y0 = c * chunkH, h = std::min( chunkH, p.bandH - y0 );
      if( filter )
      {
        for( int r = y0 / p.ctuSize; r < ( y0 + h + p.ctuSize - 1 ) / p.ctuSize; r++ )
        {
          const int ry0 = r * p.ctuSize, rh = std::min( p.ctuSize, p.bandH - ry0 );
          for( int col = 0; col < p.numCtuCols; col++ )
          {
            if( p.ctuSetIdx[r * p.numCtuCols + col] < 0 ) continue;
            const int x0 = col * p.ctuSize, wd = std::min( p.ctuSize, p.picWidth - x0 );
            for( int y = ry0; y < ry0 + rh; y++ )
            {
              memcpy( p.dst + ( ptrdiff_t ) y * p.dstStride + x0, w->hOut + ( size_t ) y * p.picWidth + x0, wd * sizeof( Pel ) );
            }
          }
        }
      }
      if( steps & DN_CLS )
      {
        const size_t n = ( size_t ) clsStride * ( ( h + 3 ) >> 2 ), off = ( size_t ) ( y0 >> 2 ) * clsStride;
        memcpy( p.outClass + off, w->hCls + off, n );
      }
      unpackMs += ms( a, clk::now() );
    }
    const auto t3 = clk::now();

    // ---- event timing: per-stage sums over chunks + overall GPU span
    float meta = 0;
    cudaEventElapsedTime( &meta, w->evMeta0, w->evMeta1 );
    double sH2D = meta, sCls = 0, sFlt = 0, sD2H = 0, span = 0;
    for( int c = 0; c < nChunks; c++ )
    {
      float a, b, d, e, f;
      cudaEventElapsedTime( &a, EV( c, 0 ), EV( c, 1 ) );
      cudaEventElapsedTime( &b, EV( c, 2 ), EV( c, 3 ) );
      cudaEventElapsedTime( &d, EV( c, 3 ), EV( c, 4 ) );
      cudaEventElapsedTime( &e, EV( c, 4 ), EV( c, 5 ) );
      cudaEventElapsedTime( &f, w->evMeta0, EV( c, 5 ) );
      sH2D += a; sCls += b; sFlt += d; sD2H += e; span = std::max<double>( span, f );
    }

    t.hostPackMs   += packMs;
    t.h2dMs        += sH2D;
    t.classifyMs   += sCls;
    t.filterMs     += sFlt;
    t.d2hMs        += sD2H;
    t.hostUnpackMs += unpackMs;
    t.gpuSpanMs    += span;
    t.totalMs      += ms( t0, t3 );
    t.chunks       += nChunks;
    t.bytesH2D     += h2d;
    t.bytesD2H     += d2h;
  }
};

Backend* createCudaBackend()
{
  return new CudaBackend;   // init() reports failure when no device is available
}

} // namespace alfgpu
