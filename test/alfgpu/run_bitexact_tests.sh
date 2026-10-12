#!/usr/bin/env bash
# Bit-exactness regression for --AlfGpu: encodes each configuration with the original CPU ALF (AlfGpu=0)
# and with AlfGpu=1 (GPU classification+filter), AlfGpu=2 (GPU filter) and AlfGpu=3 (verify), then compares
# bitstream and reconstruction MD5s. Without a CUDA build (or with VVENC_ALF_GPU_EMU=1) the host emulation
# backend runs the same data flow on the CPU.
#
# usage: run_bitexact_tests.sh <vvencFFapp> <seq_dir> <out_dir>
set -u
APP=${1:-bin/release-static/vvencFFapp}
SEQ=${2:-seq}
OUT=${3:-alfgpu_tests}
mkdir -p "$OUT"

# name | input | size | extra options
CONFIGS=(
  "ra_416_ctu128_t0|s416x240.yuv|416x240|--preset faster -f 8 -q 32 -t 0"
  "ra_416_ctu64_t0|s416x240.yuv|416x240|--preset fast -f 8 -q 27 -t 0 --CTUSize 64"
  "ra_416_nonlin_t0|s416x240.yuv|416x240|--preset faster -f 8 -q 37 -t 0 --UseNonLinearAlfLuma 1"
  "ra_416_nonlinYC_t0|s416x240.yuv|416x240|--preset faster -f 8 -q 32 -t 0 --UseNonLinearAlfLuma 1 --UseNonLinearAlfChroma 1"
  "ra_416_medium_t1|s416x240.yuv|416x240|--preset medium -f 4 -q 32 -t 1"
  "ai_832x472_t2|s832x472.yuv|832x472|--preset faster -f 4 -q 30 -t 2 -ip 1"
  "ra_832x472_ifp_t4|s832x472.yuv|832x472|--preset faster -f 6 -q 32 -t 4 --IFP 1"
  "ra_832x472_tiles_t2|s832x472.yuv|832x472|--preset faster -f 6 -q 32 -t 2 --Tiles 2x1"
  "ra_1080p_t2|s1920x1080.yuv|1920x1080|--preset faster -f 4 -q 32 -t 2"
)

fail=0
printf "%-24s %-8s %-34s %-34s %s\n" config mode bitstream_md5 recon_md5 result
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r name inp size opts <<< "$c"
  ref_bs=""; ref_rec=""
  for mode in 0 1 2 3; do
    base="$OUT/${name}_m$mode"
    # shellcheck disable=SC2086
    "$APP" -i "$SEQ/$inp" -s "$size" -fr 30 --InputBitDepth 8 --InternalBitDepth 10 $opts \
           --AlfGpu $mode -b "$base.266" -o "$base.yuv" > "$base.log" 2>&1
    if [ $? -ne 0 ]; then echo "$name mode $mode: ENCODER FAILED (see $base.log)"; fail=1; continue; fi
    bs=$(md5sum < "$base.266" | cut -c1-32); rec=$(md5sum < "$base.yuv" | cut -c1-32)
    res="ref"
    if [ $mode -eq 0 ]; then ref_bs=$bs; ref_rec=$rec
    else
      if [ "$bs" == "$ref_bs" ] && [ "$rec" == "$ref_rec" ]; then res="IDENTICAL"; else res="DIFFERENT"; fail=1; fi
      if [ $mode -eq 3 ]; then
        v=$(grep "VERIFY:" "$base.log" | tail -1)
        case "$v" in *"-> BIT-EXACT"*) res="$res, verify BIT-EXACT";; *) res="$res, verify: ${v:-missing}"; fail=1;; esac
      fi
    fi
    printf "%-24s %-8s %-34s %-34s %s\n" "$name" "$mode" "$bs" "$rec" "$res"
  done
  rm -f "$OUT/${name}"_m*.yuv
done
backend=$(grep -h "backend:" "$OUT"/*_m1.log | head -1)
echo "${backend}"
[ $fail -eq 0 ] && echo "ALL CONFIGURATIONS BIT-EXACT" || echo "SOME CONFIGURATIONS FAILED"
exit $fail
