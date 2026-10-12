/* -----------------------------------------------------------------------------
test_alf_core.cpp - compares the shared ALF core (AlfGpuCore.h) against VVenC's
own classification and 7x7 luma filter (scalar AND SIMD versions) on synthetic
10-bit pictures, for several CTU sizes, contents and random filter sets.

Build (from the vvenc root, after building libvvenc):
  g++ -O2 -std=gnu++14 -flto=auto -DTARGET_SIMD_X86=1 -DHAVE_PTHREADS \
      -Iinclude -Ibuild/base -Isource/Lib -Isource/Lib/CommonLib -Isource/Lib/CommonLib/x86 \
      test/alfgpu/test_alf_core.cpp lib/release-static/libvvenc.a -lpthread -o test_alf_core
------------------------------------------------------------------------------- */

#include "CommonLib/AdaptiveLoopFilter.h"
#include "CommonLib/AlfGpuCore.h"

#include <cstdio>
#include <random>
#include <vector>

using namespace vvenc;

static void fillPicture( std::vector<Pel>& buf, int stride, int margin, int W, int H, int content, std::mt19937& rng )
{
  Pel* p = &buf[margin * stride + margin];
  std::uniform_int_distribution<int> u( 0, 1023 );
  for( int y = 0; y < H; y++ )
    for( int x = 0; x < W; x++ )
    {
      int v = 0;
      switch( content )
      {
      case 0: v = u( rng ); break;                                                        // white noise, extremes
      case 1: v = ( x * 7 + y * 3 ) % 1024; break;                                        // gradients / ramps
      case 2: v = ( ( x / 9 + y / 5 ) & 1 ) ? 1023 : 0; break;                             // hard edges
      case 3: v = std::min( 1023, std::max( 0, 512 + ( int ) ( 300 * std::sin( x * 0.07 ) * std::cos( y * 0.05 ) ) + u( rng ) % 40 - 20 ) ); break;
      default: v = ( ( x ^ y ) & 8 ) ? u( rng ) : ( x * 4 ) % 1024; break;               // mixed texture
      }
      p[y * stride + x] = ( Pel ) v;
    }
  // replicate padding (VVenC extendBorderPel)
  for( int y = 0; y < H; y++ )
    for( int k = 1; k <= margin; k++ )
    {
      p[y * stride - k]         = p[y * stride];
      p[y * stride + W - 1 + k] = p[y * stride + W - 1];
    }
  for( int k = 1; k <= margin; k++ )
  {
    memcpy( p - k * stride - margin, p - margin, ( W + 2 * margin ) * sizeof( Pel ) );
    memcpy( p + ( H - 1 + k ) * stride - margin, p + ( H - 1 ) * stride - margin, ( W + 2 * margin ) * sizeof( Pel ) );
  }
}

int main()
{
  std::mt19937 rng( 1234 );
  const int bitDepth = 10, margin = 8;
  const int clipVals[4] = { 1024, 128, 32, 8 };

  AdaptiveLoopFilter alfSimd( true ), alfScalar( false );

  long long blocks = 0, samples = 0, errCls = 0, errFlt = 0;

  struct Cfg { int W, H, ctu; };
  const Cfg cfgs[] = { { 416, 240, 64 }, { 416, 240, 128 }, { 832, 480, 128 }, { 256, 264, 64 }, { 1920, 1080, 128 } };

  for( const Cfg& cfg : cfgs )
  for( int content = 0; content < 5; content++ )
  for( int useSimd = 0; useSimd < 2; useSimd++ )
  for( int nonLin = 0; nonLin < 2; nonLin++ )
  {
    AdaptiveLoopFilter& alf = useSimd ? alfSimd : alfScalar;
    const int W = cfg.W, H = cfg.H, S = cfg.ctu, vbH = S, vbPos = S - 4;
    const int stride = W + 2 * margin;
    std::vector<Pel> src( ( H + 2 * margin ) * stride ), dstRef( ( H + 2 * margin ) * stride, 0 );
    fillPicture( src, stride, margin, W, H, content, rng );
    const Pel* s = &src[margin * stride + margin];
    Pel* dRef    = &dstRef[margin * stride + margin];

    // random filter set: 25 classes x 13 coefficients
    std::vector<short> coef( 325 ), clip( 325 );
    std::uniform_int_distribution<int> uc( -128, 127 ), uk( 0, 3 );
    for( int c = 0; c < 25; c++ )
    {
      for( int k = 0; k < 12; k++ )
      {
        coef[c * 13 + k] = ( short ) ( content == 0 ? uc( rng ) : uc( rng ) / 4 );
        clip[c * 13 + k] = ( short ) ( nonLin ? clipVals[uk( rng )] : 1024 );
      }
      coef[c * 13 + 12] = 128; clip[c * 13 + 12] = 1024;
    }

    ClpRng clpRng; clpRng.bd = bitDepth;
    const CPelBuf  srcBuf( s, stride, W, H );
    const CPelUnitBuf srcUnit( VVENC_CHROMA_400, srcBuf );
    const PelUnitBuf  dstUnit( VVENC_CHROMA_400, PelBuf( dRef, stride, W, H ) );
    alignas( 64 ) char dummyCs[64] = { 0 };   // filterBlk takes a CodingStructure& it does not use for luma
    const CodingStructure& cs = *reinterpret_cast<const CodingStructure*>( dummyCs );

    std::vector<AlfClassifier> cls( 32 * 32 );
    for( int y0 = 0; y0 < H; y0 += S )
    for( int x0 = 0; x0 < W; x0 += S )
    {
      const int w = std::min( S, W - x0 ), h = std::min( S, H - y0 );
      const Area blk( x0, y0, w, h );
      // VVenC classification (shift = bitDepth + 4)
      alf.m_deriveClassificationBlk( cls.data(), srcBuf, blk, blk, bitDepth + 4, vbH, vbPos );
      // VVenC filter
      alf.m_filter7x7Blk[nonLin]( cls.data(), dstUnit, srcUnit, blk, blk, COMP_Y, coef.data(), clip.data(), clpRng, cs, vbH, vbPos );

      for( int by = 0; by < h; by += 4 )
      for( int bx = 0; bx < w; bx += 4 )
      {
        int c, t;
        alfgpu::classifyBlock4x4( s + ( y0 + by ) * stride + x0 + bx, stride, y0 + by, bitDepth + 4, vbH, vbPos, c, t );
        const AlfClassifier& r = cls[( by / 4 ) * 32 + bx / 4];
        blocks++;
        if( c != r.classIdx || t != r.transposeIdx )
        {
          if( errCls++ < 10 ) printf( "CLS mismatch simd=%d W=%d ctu=%d content=%d (%d,%d): core=(%d,%d) vvenc=(%d,%d)\n",
                                      useSimd, W, S, content, x0 + bx, y0 + by, c, t, r.classIdx, r.transposeIdx );
        }
        for( int yy = 0; yy < 4; yy++ )
        for( int xx = 0; xx < 4; xx++ )
        {
          const int x = x0 + bx + xx, y = y0 + by + yy;
          const Pel v = alfgpu::filterLumaSample( s + y * stride + x, stride, y, &coef[r.classIdx * 13], &clip[r.classIdx * 13],
                                                  r.transposeIdx, vbH, vbPos, ( 1 << bitDepth ) - 1 );
          samples++;
          if( v != dRef[y * stride + x] )
          {
            if( errFlt++ < 10 ) printf( "FLT mismatch simd=%d nl=%d W=%d ctu=%d content=%d (%d,%d): core=%d vvenc=%d\n",
                                        useSimd, nonLin, W, S, content, x, y, v, dRef[y * stride + x] );
          }
        }
      }
    }
  }

  printf( "classification: %lld blocks, %lld mismatches\n", blocks, errCls );
  printf( "filter        : %lld samples, %lld mismatches\n", samples, errFlt );
  printf( ( errCls || errFlt ) ? "RESULT: NOT bit-exact\n" : "RESULT: bit-exact with VVenC (scalar and SIMD)\n" );
  return ( errCls || errFlt ) ? 1 : 0;
}
