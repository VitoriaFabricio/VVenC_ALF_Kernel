/* -----------------------------------------------------------------------------
AlfGpuKernels.cuh - CUDA kernels for bit-exact ALF luma (classification + filter)

The kernel bodies are the phase functions of AlfGpuTiles.h separated by
__syncthreads(); the host emulation backend runs the very same phases, so the
tiling and indexing below are validated on the CPU as well.
Shared by the VVenC backend (AlfGpuCuda.cu) and the standalone benchmark (internal linkage:
both translation units can be linked into one program).
------------------------------------------------------------------------------- */
#pragma once

#include "../AlfGpuTiles.h"
#include "../AlfGpuCov.h"
#include <cuda_runtime.h>

namespace alfgpu
{

// one block = 128 x 32 samples = 256 blocks of 4x4, one thread each; laplacian cells shared in smem
static __global__ void __launch_bounds__( CT_THREADS ) kAlfClassify( const Pel* __restrict__ d_in, const int16_t* __restrict__ d_ctuSet,
                                                              uint8_t* __restrict__ d_cls, KParams k )
{
  __shared__ ClsSmem s;
  const int x0 = blockIdx.x * CT_W, y0 = blockIdx.y * CT_H;
  clsPhaseLoad( s, threadIdx.x, d_in, k, x0, y0 );
  __syncthreads();
  clsPhaseCells( s, threadIdx.x, k, y0 );
  __syncthreads();
  clsPhaseBlocks( s, threadIdx.x, d_ctuSet, d_cls, k, x0, y0 );
}

// one block = 32 x 32 samples, 32 x 8 threads, 4 samples per thread
static __global__ void __launch_bounds__( FT_THREADS ) kAlfFilterLuma( const Pel* __restrict__ d_in, const int16_t* __restrict__ d_ctuSet,
                                                                const uint8_t* __restrict__ d_cls,
                                                                const int16_t* __restrict__ d_coef, const int16_t* __restrict__ d_clip,
                                                                Pel* __restrict__ d_out, KParams k )
{
  __shared__ FltSmem s;
  const int x0 = blockIdx.x * FT_W, y0 = blockIdx.y * FT_H;
  fltPhaseLoad( s, threadIdx.y * FT_TX + threadIdx.x, d_in, k, x0, y0 );
  __syncthreads();
  fltPhaseCompute( s, threadIdx.x, threadIdx.y, d_ctuSet, d_cls, d_coef, d_clip, d_out, k, x0, y0 );
}

// covariance statistics: grid (units, element groups); thread = one element of one unit; the threads of a block
// compute the ELocal vectors of CV_NB 4x4 blocks together, then each adds its element of every block in CPU order
static __global__ void __launch_bounds__( CV_THREADS ) kAlfCovariance( CovKParams k, const Pel* __restrict__ rec, const Pel* __restrict__ org,
                                                                      const uint8_t* __restrict__ cls, const int* __restrict__ unitStart,
                                                                      const int16_t* __restrict__ ctuRect, const uint32_t* __restrict__ elem,
                                                                      float* __restrict__ out, uint8_t* __restrict__ used )
{
  __shared__ CovSmem s;
  const int unit = blockIdx.x, tid = threadIdx.x, e = blockIdx.y * CV_THREADS + tid;
  const bool active = e < k.numElem;
  const uint32_t ed = active ? elem[e] : 0;
  float acc[CV_MAXCLS];
  for( int c = 0; c < k.numClasses; c++ ) acc[c] = 0.f;
  uint32_t usedMask = 0;
  for( int ctu = unitStart[unit]; ctu < unitStart[unit + 1]; ctu++ )
  {
    const int x0 = ctuRect[4 * ctu], y0 = ctuRect[4 * ctu + 1], bw = ctuRect[4 * ctu + 2] >> 2, bh = ctuRect[4 * ctu + 3] >> 2;
    for( int q0 = 0; q0 < bw * bh; q0 += CV_NB )
    {
      const int nb = min( CV_NB, bw * bh - q0 );
      covPhaseLoad( s, tid, k, rec, org, cls, x0, y0, bw, q0, nb );
      __syncthreads();
      if( active ) covPhaseAccum( s, ed, nb, acc, usedMask );
      __syncthreads();
    }
  }
  if( active )
  {
    for( int c = 0; c < k.numClasses; c++ ) out[( ( size_t ) unit * k.numClasses + c ) * k.numElem + e] = acc[c];
  }
  if( blockIdx.y == 0 && tid == 0 )
  {
    for( int c = 0; c < k.numClasses; c++ ) used[unit * k.numClasses + c] = ( usedMask >> c ) & 1;
  }
}

static inline void launchAlfCovariance( const CovKParams& k, int numUnits, const Pel* rec, const Pel* org, const uint8_t* cls,
                                        const int* unitStart, const int16_t* ctuRect, const uint32_t* elem, float* out, uint8_t* used,
                                        cudaStream_t st )
{
  const dim3 grid( numUnits, ( k.numElem + CV_THREADS - 1 ) / CV_THREADS );
  kAlfCovariance<<<grid, CV_THREADS, 0, st>>>( k, rec, org, cls, unitStart, ctuRect, elem, out, used );
}

static inline dim3 clsGrid( const KParams& k ) { return dim3( ( k.picWidth + CT_W - 1 ) / CT_W, ( k.bandH + CT_H - 1 ) / CT_H ); }
static inline dim3 fltGrid( const KParams& k ) { return dim3( ( k.picWidth + FT_W - 1 ) / FT_W, ( k.bandH + FT_H - 1 ) / FT_H ); }

static inline void launchAlfClassify( const Pel* d_in, const int16_t* d_ctuSet, uint8_t* d_cls, const KParams& k, cudaStream_t s )
{
  kAlfClassify<<<clsGrid( k ), CT_THREADS, 0, s>>>( d_in, d_ctuSet, d_cls, k );
}

static inline void launchAlfFilterLuma( const Pel* d_in, const int16_t* d_ctuSet, const uint8_t* d_cls, const int16_t* d_coef,
                                 const int16_t* d_clip, Pel* d_out, const KParams& k, cudaStream_t s )
{
  kAlfFilterLuma<<<fltGrid( k ), dim3( FT_TX, FT_TY ), 0, s>>>( d_in, d_ctuSet, d_cls, d_coef, d_clip, d_out, k );
}

} // namespace alfgpu
