/* -----------------------------------------------------------------------------
AlfGpuTiles.h - tiled, massively parallel ALF luma: the phases of each CUDA block

Each CUDA kernel is written as a sequence of phases separated by __syncthreads():
    phase(shared memory, thread index, ...)
The same phase functions are executed
  * on the GPU by the kernels in cuda/AlfGpuKernels.cuh (one call per thread), and
  * on the CPU by the host simulator in AlfGpuHost.cpp (loop over blocks and threads),
so the emulation backend validates the exact tiling / indexing of the CUDA code.

Classification kernel (256 threads, tile of 128 x 32 samples = 32 x 8 blocks of 4x4):
  1. load   : 134 x 38 samples (3-sample halo) into shared memory, coalesced
  2. cells  : every 2x2 laplacian cell of the tile is computed ONCE (66 x 18 cells,
              ~4.6 per thread) and stored in shared memory
  3. blocks : one thread per 4x4 block sums its 4x4 (or 3x4 at the VB) cells
Neighbouring 4x4 blocks share 3/4 of their cells, so phase 2 removes ~3.5x of the
laplacian arithmetic of the per-block version and makes the work uniform per thread.

Filter kernel (32 x 8 threads, tile of 32 x 32 samples, 4 samples per thread):
  1. load    : 38 x 38 samples into shared memory
  2. compute : 7x7 diamond filter per sample from shared memory
------------------------------------------------------------------------------- */
#pragma once

#include "AlfGpuCore.h"

#include <cstdlib>

namespace alfgpu
{

struct KParams
{
  int picWidth, bandY0, bandH, ctuSize, numCtuCols, bitDepth, vbH, vbPos;
  int inPitch;     // samples per row of the input band (picWidth + 2*kHalo)
  int clsStride;   // bytes per row of 4x4 blocks (= picWidth / 4)
  int outPitch;    // samples per row of the output
};

// ------------------------------------------------------------------ classification
static constexpr int CT_THREADS = 256;
static constexpr int CT_W  = 128, CT_H = 32;                   // samples per tile
static constexpr int CT_BX = CT_W / 4, CT_BY = CT_H / 4;       // 32 x 8 blocks per tile (= 256 threads)
static constexpr int CT_PW = CT_W + 6, CT_PH = CT_H + 6;       // sample tile with 3-sample halo
static constexpr int CT_CW = CT_W / 2 + 2, CT_CH = CT_H / 2 + 2;   // cells: c = x0-2 .. x0+CT_W, r = y0-2 .. y0+CT_H (step 2)
static_assert( CT_BX * CT_BY == CT_THREADS, "one thread per 4x4 block" );

struct ClsSmem
{
  Pel pix[CT_PH][CT_PW];
  int cv [CT_CH][CT_CW];
  int ch [CT_CH][CT_CW];
  int cd0[CT_CH][CT_CW];
  int cd1[CT_CH][CT_CW];
};

// samples of the band outside [-kHalo, W+kHalo) x [-kHalo, bandH+kHalo) are not readable: they are
// replaced by 0 and only ever feed 4x4 blocks / samples outside the picture, which are discarded
ALFG_HD Pel loadBandSample( const Pel* d_in, const KParams& k, int gx, int gy )
{
  if( gx < -kHalo || gx >= k.picWidth + kHalo || gy < -kHalo || gy >= k.bandH + kHalo ) return 0;
  return d_in[( ptrdiff_t ) ( gy + kHalo ) * k.inPitch + ( gx + kHalo )];
}

ALFG_HD void clsPhaseLoad( ClsSmem& s, int tid, const Pel* d_in, const KParams& k, int x0, int y0 )
{
  for( int i = tid; i < CT_PW * CT_PH; i += CT_THREADS )
  {
    const int r = i / CT_PW, c = i - r * CT_PW;
    s.pix[r][c] = loadBandSample( d_in, k, x0 - 3 + c, y0 - 3 + r );
  }
}

ALFG_HD void clsPhaseCells( ClsSmem& s, int tid, const KParams& k, int y0 )
{
  for( int i = tid; i < CT_CW * CT_CH; i += CT_THREADS )
  {
    const int ci = i / CT_CW, cj = i - ci * CT_CW;
    const int r  = y0 - 2 + 2 * ci;                     // band row of the cell
    // cell (r, c) is pixel tile entry [r - (y0-3)][c - (x0-3)] = [2ci+1][2cj+1]
    int v, h, d0, d1;
    lapCell( &s.pix[2 * ci + 1][2 * cj + 1], CT_PW, k.bandY0 + r, k.vbH, k.vbPos, v, h, d0, d1 );
    s.cv[ci][cj] = v; s.ch[ci][cj] = h; s.cd0[ci][cj] = d0; s.cd1[ci][cj] = d1;
  }
}

ALFG_HD void clsPhaseBlocks( const ClsSmem& s, int tid, const int16_t* d_ctuSet, uint8_t* d_cls, const KParams& k, int x0, int y0 )
{
  const int tx = tid % CT_BX, ty = tid / CT_BX;
  const int bx = x0 + 4 * tx, by = y0 + 4 * ty;          // band coordinates of the block
  if( bx >= k.picWidth || by >= k.bandH ) return;
  if( d_ctuSet[( by / k.ctuSize ) * k.numCtuCols + bx / k.ctuSize] < 0 ) return;

  const int y = k.bandY0 + by;
  int kBeg, kEnd;
  cellRowRange( y, k.vbH, k.vbPos, kBeg, kEnd );
  int sumV = 0, sumH = 0, sumD0 = 0, sumD1 = 0;
  for( int kk = kBeg; kk < kEnd; kk++ )
  {
    const int ci = 2 * ty + kk;                          // r = by - 2 + 2kk
    for( int l = 0; l < 4; l++ )
    {
      const int cj = 2 * tx + l;                         // c = bx - 2 + 2l
      sumV += s.cv[ci][cj]; sumH += s.ch[ci][cj]; sumD0 += s.cd0[ci][cj]; sumD1 += s.cd1[ci][cj];
    }
  }
  int c, t;
  classFromSums( sumV, sumH, sumD0, sumD1, y, k.bitDepth + 4, k.vbH, k.vbPos, c, t );
  d_cls[( by >> 2 ) * k.clsStride + ( bx >> 2 )] = ( uint8_t ) ( c | ( t << 5 ) );
}

// ------------------------------------------------------------------ filter
static constexpr int FT_TX = 32, FT_TY = 8, FT_ROWS = 4;
static constexpr int FT_THREADS = FT_TX * FT_TY;
static constexpr int FT_W  = FT_TX, FT_H = FT_TY * FT_ROWS;    // 32 x 32 samples per tile
static constexpr int FT_PW = FT_W + 6, FT_PH = FT_H + 6;

struct FltSmem
{
  Pel pix[FT_PH][FT_PW];
};

ALFG_HD void fltPhaseLoad( FltSmem& s, int tid, const Pel* d_in, const KParams& k, int x0, int y0 )
{
  for( int i = tid; i < FT_PW * FT_PH; i += FT_THREADS )
  {
    const int r = i / FT_PW, c = i - r * FT_PW;
    s.pix[r][c] = loadBandSample( d_in, k, x0 - 3 + c, y0 - 3 + r );
  }
}

ALFG_HD void fltPhaseCompute( const FltSmem& s, int tx, int ty, const int16_t* d_ctuSet, const uint8_t* d_cls,
                              const int16_t* d_coef, const int16_t* d_clip, Pel* d_out, const KParams& k, int x0, int y0 )
{
  const int x = x0 + tx;
  if( x >= k.picWidth ) return;
  const int maxVal = ( 1 << k.bitDepth ) - 1;
  for( int j = 0; j < FT_ROWS; j++ )
  {
    const int ly = ty + j * FT_TY;
    const int y  = y0 + ly;
    if( y >= k.bandH ) break;
    const int set = d_ctuSet[( y / k.ctuSize ) * k.numCtuCols + x / k.ctuSize];
    if( set < 0 ) continue;
    const uint8_t cl   = d_cls[( y >> 2 ) * k.clsStride + ( x >> 2 )];
    const int classIdx = cl & 31, tr = cl >> 5;
    const int off      = set * kSetStride + classIdx * kNumCoeff;
    d_out[( ptrdiff_t ) y * k.outPitch + x] =
      filterLumaSample( &s.pix[ly + 3][tx + 3], FT_PW, k.bandY0 + y, d_coef + off, d_clip + off, tr, k.vbH, k.vbPos, maxVal );
  }
}

// ------------------------------------------------------------------ chunking of a band
// A band (one CTU row or the whole frame) is split into chunks of whole CTU rows that are
// transferred / filtered / read back in a pipeline over several CUDA streams.
// env VVENC_ALF_GPU_CHUNK_ROWS = CTU rows per chunk (0 or unset: automatic, ~4 chunks per band)
inline int chunkCtuRows( int bandH, int ctuSize )
{
  const int ctuRows = ( bandH + ctuSize - 1 ) / ctuSize;
  if( const char* e = getenv( "VVENC_ALF_GPU_CHUNK_ROWS" ) )
  {
    const int v = atoi( e );
    if( v > 0 ) return v < ctuRows ? v : ctuRows;
  }
  if( ctuRows <= 2 ) return ctuRows;
  return ( ctuRows + 3 ) / 4;
}

} // namespace alfgpu
