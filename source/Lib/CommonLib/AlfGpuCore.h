/* -----------------------------------------------------------------------------
AlfGpuCore.h  -  bit-exact ALF luma core shared by CPU (host) and CUDA (device)

Every function here reproduces, sample by sample, the integer arithmetic of
  AdaptiveLoopFilter::deriveClassificationBlk()   (classification, 4x4 blocks)
  AdaptiveLoopFilter::filterBlk<ALF_FILTER_7>()    (7x7 diamond luma filter)
of VVenC, including the ALF virtual boundary (VB) handling.

The functions take a pointer to the sample being processed plus the row stride,
so they work the same on a full picture buffer, a padded band copied to the GPU,
or a tile staged in CUDA shared memory. The *global* (picture) row y is passed
separately because the VB rules depend on (y mod CTU height).

No VVenC headers are included, so this file compiles with nvcc as well as g++.
------------------------------------------------------------------------------- */
#pragma once

#include <cstdint>
#include <cstddef>

#if defined( __CUDACC__ )
#  define ALFG_HD __host__ __device__ __forceinline__
#else
#  define ALFG_HD inline
#endif

namespace alfgpu
{

typedef int16_t Pel;

static constexpr int kNumClasses   = 25;
static constexpr int kNumCoeff     = 13;                        // 12 taps + centre (128)
static constexpr int kSetStride    = kNumClasses * kNumCoeff;   // 325 shorts per filter set
static constexpr int kNumFixedSets = 16;                        // NUM_FIXED_FILTER_SETS
static constexpr int kMaxAps       = 8;                         // ALF_CTB_MAX_NUM_APS
static constexpr int kMaxSets      = kNumFixedSets + kMaxAps;   // setIdx as in m_alfCtbFilterIndex
static constexpr int kHalo         = 4;                         // samples kept around a band (>= 3 needed)
static constexpr int kShift        = 7;                         // m_NUM_BITS - 1

ALFG_HD int iabs_ ( int v )                 { return v < 0 ? -v : v; }
ALFG_HD int clip3_( int lo, int hi, int v ) { return v < lo ? lo : ( v > hi ? hi : v ); }

// th[16] = { 0, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4 }
ALFG_HD int activityToClass( int a )
{
  return a == 0 ? 0 : a == 1 ? 1 : a < 7 ? 2 : a < 15 ? 3 : 4;
}

// transposeTable[8] = { 0, 1, 0, 2, 2, 3, 1, 3 }, packed 2 bits per entry
ALFG_HD int transposeLut( int idx )
{
  return ( 0xDE84 >> ( 2 * idx ) ) & 3;
}

// ---------------------------------------------------------------------------
// Laplacian "cell": the 4 directional activities of the 2x2 cell whose top-left
// sample is (r, c). Each cell contributes |lap(r,c)| + |lap(r+1,c+1)| per direction
// (VVenC sub-sampling). A 4x4 block sums the 4x4 cells r = y-2, y, y+2, y+4 and
// c = x-2, x, x+2, x+4; neighbouring blocks share cells, so a GPU tile computes each
// cell once and all blocks of the tile reuse it.
//   p  : pointer to sample (r, c)        ry : global picture row r (even)
// Needs rows r-1 .. r+2 and columns c-1 .. c+2.
// ---------------------------------------------------------------------------
ALFG_HD void lapCell( const Pel* p, ptrdiff_t stride, int ry, int vbH, int vbPos, int& v, int& h, int& d0, int& d1 )
{
  const int vbMask = vbH - 1;
  const Pel* s1 = p;
  const Pel* s0 = s1 - stride;
  const Pel* s2 = s1 + stride;
  const Pel* s3 = s1 + 2 * stride;
  if( ry > 0 && ( ry & vbMask ) == vbPos - 2 )      s3 = s2;   // VB: no access below the boundary
  else if( ry > 0 && ( ry & vbMask ) == vbPos )     s0 = s1;   // VB: no access above the boundary

  const int y0   = s1[0] * 2;
  const int yup1 = s2[1] * 2;
  v  = iabs_( y0 - s0[0]  - s2[0] ) + iabs_( yup1 - s1[1] - s3[1] );
  h  = iabs_( y0 - s1[1]  - s1[-1]) + iabs_( yup1 - s2[2] - s2[0] );
  d0 = iabs_( y0 - s0[-1] - s2[1] ) + iabs_( yup1 - s1[0] - s3[2] );
  d1 = iabs_( y0 - s2[-1] - s0[1] ) + iabs_( yup1 - s3[0] - s1[2] );
}

// which of the 4 cell rows a block uses (VB: blocks touching the boundary use 3)
ALFG_HD void cellRowRange( int y, int vbH, int vbPos, int& kBeg, int& kEnd )
{
  const int yb = y & ( vbH - 1 );
  kBeg = 0; kEnd = 4;
  if( yb == vbPos - 4 )  kEnd = 3;   // block just above the VB: drop last pair of rows
  else if( yb == vbPos ) kBeg = 1;   // block just below the VB: drop first pair of rows
}

// class and transpose index of a 4x4 block from its summed directional activities
//   y : global picture row of the block      shift : internal bit depth + 4
ALFG_HD void classFromSums( int sumV, int sumH, int sumD0, int sumD1, int y, int shift, int vbH, int vbPos,
                            int& classIdx, int& transposeIdx )
{
  const int yb       = y & ( vbH - 1 );
  const int tempAct  = sumV + sumH;
  const int mul      = ( yb == vbPos - 4 || yb == vbPos ) ? 96 : 64;
  const int activity = clip3_( 0, 15, ( tempAct * mul ) >> shift );
  int cls            = activityToClass( activity );

  int hv1, hv0, d1, d0, dirTempHV, dirTempD;
  if( sumV > sumH )   { hv1 = sumV;  hv0 = sumH;  dirTempHV = 1; }
  else                { hv1 = sumH;  hv0 = sumV;  dirTempHV = 3; }
  if( sumD0 > sumD1 ) { d1 = sumD0;  d0 = sumD1;  dirTempD  = 0; }
  else                { d1 = sumD1;  d0 = sumD0;  dirTempD  = 2; }

  int hvd1, hvd0, mainDirection, secondaryDirection;
  // NOTE: 32-bit unsigned products, exactly as VVenC (do NOT widen to 64 bit)
  if( ( uint32_t ) d1 * ( uint32_t ) hv0 > ( uint32_t ) hv1 * ( uint32_t ) d0 )
  {
    hvd1 = d1;  hvd0 = d0;  mainDirection = dirTempD;   secondaryDirection = dirTempHV;
  }
  else
  {
    hvd1 = hv1; hvd0 = hv0; mainDirection = dirTempHV;  secondaryDirection = dirTempD;
  }

  int directionStrength = 0;
  if( hvd1 > 2 * hvd0 )     directionStrength = 1;
  if( hvd1 * 2 > 9 * hvd0 ) directionStrength = 2;

  if( directionStrength )
  {
    cls += ( ( ( mainDirection & 0x1 ) << 1 ) + directionStrength ) * 5;
  }

  classIdx     = cls;
  transposeIdx = transposeLut( mainDirection * 2 + ( secondaryDirection >> 1 ) );
}

// ---------------------------------------------------------------------------
// Classification of one 4x4 block (reference form: computes its own 16 cells).
//   p      : pointer to the top-left sample of the block
//   y      : global picture row of the block (multiple of 4)
//   shift  : internal bit depth + 4
//   vbH    : m_alfVBLumaCTUHeight (CTU height, power of 2)
//   vbPos  : m_alfVBLumaPos (CTU height - 4)
// Needs 3 readable samples above/left and 3 below/right of the block.
// ---------------------------------------------------------------------------
ALFG_HD void classifyBlock4x4( const Pel* p, ptrdiff_t stride, int y, int shift, int vbH, int vbPos,
                               int& classIdx, int& transposeIdx )
{
  int kBeg, kEnd;
  cellRowRange( y, vbH, vbPos, kBeg, kEnd );
  int sumV = 0, sumH = 0, sumD0 = 0, sumD1 = 0;
  for( int k = kBeg; k < kEnd; k++ )
  {
    for( int l = 0; l < 4; l++ )
    {
      int v, h, d0, d1;
      lapCell( p + ( ptrdiff_t ) ( 2 * k - 2 ) * stride + ( 2 * l - 2 ), stride, y - 2 + 2 * k, vbH, vbPos, v, h, d0, d1 );
      sumV += v; sumH += h; sumD0 += d0; sumD1 += d1;
    }
  }
  classFromSums( sumV, sumH, sumD0, sumD1, y, shift, vbH, vbPos, classIdx, transposeIdx );
}

ALFG_HD int clipALF( int clip, int ref, int v0, int v1 )
{
  return clip3_( -clip, clip, v0 - ref ) + clip3_( -clip, clip, v1 - ref );
}

// ---------------------------------------------------------------------------
// 7x7 diamond luma filter for one sample.
//   p0      : pointer to the sample (x, y)
//   y       : global picture row (for the VB rules)
//   coef    : 13 coefficients of the class (coef[12] = 128 is not used as a tap)
//   clip    : 13 clipping values of the class (actual values, e.g. 1024 for 10 bit)
//   maxVal  : (1 << bitDepth) - 1
// Needs 3 readable rows/columns around the sample.
// ---------------------------------------------------------------------------
ALFG_HD Pel filterLumaSample( const Pel* p0, ptrdiff_t stride, int y, const int16_t* coef, const int16_t* clip,
                              int transposeIdx, int vbH, int vbPos, int maxVal )
{
  int fc[12], fl[12];

  switch( transposeIdx )
  {
  case 1:
    {
      const int m[12] = { 9, 4, 10, 8, 1, 5, 11, 7, 3, 0, 2, 6 };
      for( int k = 0; k < 12; k++ ) { fc[k] = coef[m[k]]; fl[k] = clip[m[k]]; }
    }
    break;
  case 2:
    {
      const int m[12] = { 0, 3, 2, 1, 8, 7, 6, 5, 4, 9, 10, 11 };
      for( int k = 0; k < 12; k++ ) { fc[k] = coef[m[k]]; fl[k] = clip[m[k]]; }
    }
    break;
  case 3:
    {
      const int m[12] = { 9, 8, 10, 4, 3, 7, 11, 5, 1, 0, 2, 6 };
      for( int k = 0; k < 12; k++ ) { fc[k] = coef[m[k]]; fl[k] = clip[m[k]]; }
    }
    break;
  default:
    for( int k = 0; k < 12; k++ ) { fc[k] = coef[k]; fl[k] = clip[k]; }
    break;
  }

  // p1..p6 = rows y+1, y-1, y+2, y-2, y+3, y-3 (VVenC pImg1..pImg6)
  const Pel* p1 = p0 + stride;
  const Pel* p2 = p0 - stride;
  const Pel* p3 = p1 + stride;
  const Pel* p4 = p2 - stride;
  const Pel* p5 = p3 + stride;
  const Pel* p6 = p4 - stride;

  const int yVb = y & ( vbH - 1 );
  if( yVb < vbPos && yVb >= vbPos - 4 )        // above the virtual boundary
  {
    p1 = ( yVb == vbPos - 1 ) ? p0 : p1;
    p3 = ( yVb >= vbPos - 2 ) ? p1 : p3;
    p5 = ( yVb >= vbPos - 3 ) ? p3 : p5;

    p2 = ( yVb == vbPos - 1 ) ? p0 : p2;
    p4 = ( yVb >= vbPos - 2 ) ? p2 : p4;
    p6 = ( yVb >= vbPos - 3 ) ? p4 : p6;
  }
  else if( yVb >= vbPos && yVb <= vbPos + 3 )  // below the virtual boundary
  {
    p2 = ( yVb == vbPos )     ? p0 : p2;
    p4 = ( yVb <= vbPos + 1 ) ? p2 : p4;
    p6 = ( yVb <= vbPos + 2 ) ? p4 : p6;

    p1 = ( yVb == vbPos )     ? p0 : p1;
    p3 = ( yVb <= vbPos + 1 ) ? p1 : p3;
    p5 = ( yVb <= vbPos + 2 ) ? p3 : p5;
  }
  const bool nearVB = ( yVb == vbPos - 1 ) || ( yVb == vbPos );

  const int curr = p0[0];
  int sum = 0;
  sum += fc[0]  * clipALF( fl[0],  curr, p5[ 0], p6[ 0] );

  sum += fc[1]  * clipALF( fl[1],  curr, p3[+1], p4[-1] );
  sum += fc[2]  * clipALF( fl[2],  curr, p3[ 0], p4[ 0] );
  sum += fc[3]  * clipALF( fl[3],  curr, p3[-1], p4[+1] );

  sum += fc[4]  * clipALF( fl[4],  curr, p1[+2], p2[-2] );
  sum += fc[5]  * clipALF( fl[5],  curr, p1[+1], p2[-1] );
  sum += fc[6]  * clipALF( fl[6],  curr, p1[ 0], p2[ 0] );
  sum += fc[7]  * clipALF( fl[7],  curr, p1[-1], p2[+1] );
  sum += fc[8]  * clipALF( fl[8],  curr, p1[-2], p2[+2] );

  sum += fc[9]  * clipALF( fl[9],  curr, p0[+3], p0[-3] );
  sum += fc[10] * clipALF( fl[10], curr, p0[+2], p0[-2] );
  sum += fc[11] * clipALF( fl[11], curr, p0[+1], p0[-1] );

  if( !nearVB ) sum = ( sum + ( 1 << ( kShift - 1 ) ) ) >> kShift;
  else          sum = ( sum + ( 1 << ( kShift + 2 ) ) ) >> ( kShift + 3 );

  sum += curr;
  return ( Pel ) clip3_( 0, maxVal, sum );
}

} // namespace alfgpu
