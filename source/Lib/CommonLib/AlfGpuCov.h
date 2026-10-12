/* -----------------------------------------------------------------------------
AlfGpuCov.h - bit-exact ALF covariance statistics (the encoder's Wiener statistics)

Reproduces EncAdaptiveLoopFilter::getPreBlkStats() + getPreBlkStatsAccum() of VVenC:
for every 4x4 block of a CTU, in raster order,
  ELocal[b][k][s] : the 13 (luma) / 7 (chroma) filter inputs of sample s of the block for
                    clipping bin b (sum of the two symmetric neighbours, coefficient order
                    given by the transpose index, rows clipped at the ALF virtual boundary),
  yLocal[s]       : org - rec
  E[b0][b1][k][l] += float( sum_s ELocal[b1][l][s] * ELocal[b0][k][s] )   (l >= k)
  y[b0][k]        += float( sum_s ELocal[b0][k][s] * yLocal[s] )
  pixAcc          += float( sum_s yLocal[s]^2 )
into the covariance of the block's class, for a statistics unit (ASU or CTU) that is
reset to 0 and then fed CTU by CTU (raster inside the unit).

Bit-exactness: the per-block sums are integers (exact in any order); the float additions
are done by ONE thread per (unit, element) that walks the blocks in the CPU order and adds
into the accumulator of the block's class - the same sequence of IEEE float roundings as
the CPU. Parallelism: units x elements (e.g. 1509 elements per luma unit with non-linear
ALF), with the ELocal vectors of a batch of blocks computed cooperatively in shared memory.

Like AlfGpuTiles.h, each kernel phase is a function used by the CUDA kernel and by the
host simulator of the emulation backend.
------------------------------------------------------------------------------- */
#pragma once

#include "AlfGpuCore.h"

#include <vector>

namespace alfgpu
{

static constexpr int CV_THREADS  = 128;   // elements per CUDA block
static constexpr int CV_NB       = 16;    // 4x4 blocks per shared-memory batch
static constexpr int CV_MAXBINS  = 4;
static constexpr int CV_MAXCOEF  = 13;
static constexpr int CV_MAXCLS   = 25;

// element descriptor: kind (0 = E, 1 = y, 2 = pixAcc) | b0 | k | b1 | l
ALFG_HD uint32_t covElem( int kind, int b0, int k, int b1, int l ) { return ( uint32_t ) ( kind << 24 | b0 << 18 | k << 12 | b1 << 6 | l ); }
ALFG_HD int covKind( uint32_t e ) { return ( e >> 24 ) & 3; }
ALFG_HD int covB0  ( uint32_t e ) { return ( e >> 18 ) & 63; }
ALFG_HD int covK   ( uint32_t e ) { return ( e >> 12 ) & 63; }
ALFG_HD int covB1  ( uint32_t e ) { return ( e >> 6 ) & 63; }
ALFG_HD int covL   ( uint32_t e ) { return e & 63; }

// element list in the order of the CPU loops: E (b0, k, b1, l >= k), y (b0, k), pixAcc
inline std::vector<uint32_t> covElements( int numCoeff, int numBins )
{
  std::vector<uint32_t> v;
  for( int b0 = 0; b0 < numBins; b0++ )
    for( int k = 0; k < numCoeff; k++ )
      for( int b1 = 0; b1 < numBins; b1++ )
        for( int l = k; l < numCoeff; l++ ) v.push_back( covElem( 0, b0, k, b1, l ) );
  for( int b0 = 0; b0 < numBins; b0++ )
    for( int k = 0; k < numCoeff; k++ ) v.push_back( covElem( 1, b0, k, 0, 0 ) );
  v.push_back( covElem( 2, 0, 0, 0, 0 ) );
  return v;
}

struct CovKParams
{
  int numCoeff, halfLen, numBins, nonLinear, numClasses, numElem;
  int clip[CV_MAXBINS];
  int vbH, vbPos;
  int recPitch, orgPitch, clsStride;   // clsStride: luma only (bytes per row of 4x4 blocks)
};

struct CovSmem
{
  Pel     e[CV_NB][CV_MAXBINS][CV_MAXCOEF][16];
  Pel     y[CV_NB][16];
  uint8_t cls[CV_NB];
};

// ELocal of one sample (VVenC calcLinCovariance4 / calcCovariance4, one column)
ALFG_HD void covSample( Pel e[CV_MAXBINS][CV_MAXCOEF][16], int s, const Pel* r, ptrdiff_t pitch, int t, const CovKParams& k, int clipTop, int clipBot )
{
  const int curr = r[0];
  const int h = k.halfLen;
  int kk = 0;
  auto emit = [&]( int dy0, int dx0, int dy1, int dx1 )
  {
    const int v0 = r[dy0 * pitch + dx0], v1 = r[dy1 * pitch + dx1];
    if( !k.nonLinear )
    {
      e[0][kk][s] = ( Pel ) ( v0 + v1 - ( Pel ) ( curr << 1 ) );
    }
    else
    {
      for( int b = 0; b < k.numBins; b++ ) e[b][kk][s] = ( Pel ) ( clip3_( -k.clip[b], k.clip[b], v0 - curr ) + clip3_( -k.clip[b], k.clip[b], v1 - curr ) );
    }
    kk++;
  };
  auto top = [&]( int i ) { return i > clipTop ? i : clipTop; };          // clipIdx( i,  clipTopRow )
  auto bot = [&]( int i ) { return -( i > -clipBot ? i : -clipBot ); };   // -clipIdx( i, -clipBotRow )
  if( t == 0 )
  {
    for( int i = -h; i < 0; i++ )
      for( int j = -h - i; j <= h + i; j++ ) emit( top( i ), j, bot( i ), -j );
    for( int j = -h; j < 0; j++ ) emit( 0, j, 0, -j );
  }
  else if( t == 1 )
  {
    for( int j = -h; j < 0; j++ )
      for( int i = -h - j; i <= h + j; i++ ) emit( top( i ), j, bot( i ), -j );
    for( int i = -h; i < 0; i++ ) emit( top( i ), 0, bot( i ), 0 );
  }
  else if( t == 2 )
  {
    for( int i = -h; i < 0; i++ )
      for( int j = h + i; j >= -h - i; j-- ) emit( top( i ), j, bot( i ), -j );
    for( int j = -h; j < 0; j++ ) emit( 0, j, 0, -j );
  }
  else
  {
    for( int j = -h; j < 0; j++ )
      for( int i = h + j; i >= -h - j; i-- ) emit( top( i ), j, bot( i ), -j );
    for( int i = -h; i < 0; i++ ) emit( top( i ), 0, bot( i ), 0 );
  }
  for( int b = 0; b < ( k.nonLinear ? k.numBins : 1 ); b++ ) e[b][kk][s] = ( Pel ) curr;
}

// phase 1: ELocal / yLocal / class of blocks [q0, q0 + nb) of the CTU at (x0, y0) with bw blocks per row
// rec / org point to plane sample (0, 0); cls: luma class map (classIdx | transposeIdx << 5) or nullptr
ALFG_HD void covPhaseLoad( CovSmem& s, int tid, const CovKParams& k, const Pel* rec, const Pel* org, const uint8_t* cls,
                           int x0, int y0, int bw, int q0, int nb )
{
  for( int i = tid; i < nb * 16; i += CV_THREADS )
  {
    const int blk = i >> 4, smp = i & 15, ii = smp >> 2, xx = smp & 3;
    const int q = q0 + blk, by = q / bw, bx = q - by * bw;
    const int px = x0 + 4 * bx + xx, py = y0 + 4 * by + ii;
    int clipTop = -4, clipBot = 4;
    const int vbDistance = ( py % k.vbH ) - k.vbPos;
    if( vbDistance >= -3 && vbDistance < 0 ) { clipBot = -vbDistance - 1; clipTop = -clipBot; }
    else if( vbDistance >= 0 && vbDistance < 3 ) { clipTop = -vbDistance; clipBot = -clipTop; }
    int c = 0, t = 0;
    if( cls )
    {
      const uint8_t v = cls[( py >> 2 ) * k.clsStride + ( px >> 2 )];
      c = v & 31; t = v >> 5;
    }
    const Pel* r = rec + ( ptrdiff_t ) py * k.recPitch + px;
    covSample( s.e[blk], smp, r, k.recPitch, t, k, clipTop, clipBot );
    s.y[blk][smp] = ( Pel ) ( org[( ptrdiff_t ) py * k.orgPitch + px] - r[0] );
    if( smp == 0 ) s.cls[blk] = ( uint8_t ) c;
  }
}

// phase 2: one element, blocks of the batch in order, float accumulation per class (CPU order)
ALFG_HD void covPhaseAccum( const CovSmem& s, uint32_t ed, int nb, float* acc, uint32_t& usedMask )
{
  const int kind = covKind( ed ), b0 = covB0( ed ), kk = covK( ed ), b1 = covB1( ed ), l = covL( ed );
  for( int blk = 0; blk < nb; blk++ )
  {
    int sum = 0;
    if( kind == 0 )
    {
      const Pel* a = s.e[blk][b1][l];
      const Pel* b = s.e[blk][b0][kk];
      for( int i = 0; i < 16; i++ ) sum += int( a[i] ) * b[i];
    }
    else if( kind == 1 )
    {
      const Pel* a = s.e[blk][b0][kk];
      for( int i = 0; i < 16; i++ ) sum += int( a[i] ) * s.y[blk][i];
    }
    else
    {
      for( int i = 0; i < 16; i++ ) sum += int( s.y[blk][i] ) * s.y[blk][i];
    }
    const int c = s.cls[blk];
#if defined( __CUDA_ARCH__ )
    acc[c] = __fadd_rn( acc[c], __int2float_rn( sum ) );
#else
    acc[c] = acc[c] + ( float ) sum;     // VVenC: float += int (SSE cvtsi2ss / cvtdq2ps + addss / addps)
#endif
    usedMask |= 1u << c;
  }
}

} // namespace alfgpu
