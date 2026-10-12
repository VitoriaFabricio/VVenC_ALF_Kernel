# ALF luma bit-exato em CUDA integrado ao VVenC

Base: VVenC `master` @ `b401ac9` (12/10/2026). Para aplicar:

```bash
git clone https://github.com/fraunhoferhhi/vvenc.git && cd vvenc
git apply /caminho/vvenc_alf_gpu.patch   # se a sua árvore for outra versão, use git apply --3way
```

O patch adiciona:

| Arquivo | O que é |
|---|---|
| `source/Lib/CommonLib/AlfGpuCore.h` | Núcleo bit-exato (`__host__ __device__`): classificação 4x4 e filtro diamante 7x7, com fronteira virtual (VB). Usado pela CPU e pela GPU. |
| `source/Lib/CommonLib/AlfGpuTiles.h` | As **fases** de cada bloco CUDA (carga em shared memory, laplacianos por célula, soma por bloco 4x4, filtro). Executadas pela GPU e, em simulação, pela CPU. |
| `source/Lib/CommonLib/AlfGpuCov.h` | Estatísticas de covariância do codificador (as matrizes E, y das equações de Wiener), bit-exatas com `getPreBlkStats` do VVenC: fases do kernel CUDA, também simuladas na CPU. |
| `source/Lib/CommonLib/AlfGpu.h` / `AlfGpuHost.cpp` | Interface do backend + backend de emulação na CPU (simula os kernels bloco a bloco, thread a thread). |
| `source/Lib/CommonLib/cuda/AlfGpuKernels.cuh` / `AlfGpuCuda.cu` | Kernels CUDA + backend CUDA: quadro inteiro numa chamada, pipeline em pedaços sobre vários streams, eventos e NVML. |
| `source/Lib/EncoderLib/AlfProfiler.{h,cpp}` | Tempo por etapa do ALF + energia (RAPL na CPU, NVML na GPU). |
| `EncAdaptiveLoopFilter.*`, `EncSlice.cpp`, `EncLib.cpp` | Integração na etapa `ALF_RECONSTRUCT`, dump de quadros, relatório. |
| `vvencCfg.h/.cpp`, `VVEncAppCfg.h` | Opções `--AlfGpu` e `--AlfProf` (ocupam os 2 `m_reservedInt8`, ABI inalterada). |
| `test/alfgpu/` | Teste do núcleo contra o VVenC, regressão bit-exata, benchmark isolado. |

## 1. Por que o kernel original não era bit-exato

Classificação (`deriveClassificationBlk`):
1. **Janela 8x8 deslocada**: o kernel somava linhas/colunas `y..y+7`; o VVenC usa `y-2..y+5` e `x-2..x+5`.
2. **Subamostragem incompleta**: o VVenC calcula o laplaciano em duas posições de cada célula 2x2, (par,par) **e** (ímpar,ímpar): 32 amostras por direção. O kernel usava só (par,par).
3. **Shift da atividade**: é `bitDepth + 4` (14 em 10 bits); o kernel usava 12, o que multiplica a atividade por 4.
4. **Fronteira virtual (4 linhas acima do fim de cada CTU)**: o VVenC substitui linhas acima/abaixo da VB, soma só 3 pares de linhas e usa multiplicador 96 (em vez de 64) nos blocos colados à VB. Não existia no kernel.
5. **Produto da decisão de direção**: o VVenC usa `uint32_t` (com estouro); o kernel usava 64 bits. Em 10 bits o produto máximo ainda cabe em 32 bits, mas o núcleo replica o VVenC exatamente.

Filtro (`filterBlk<ALF_FILTER_7>`):

6. **Forma errada**: as tabelas `dy/dx` não formam o diamante 7x7. As 12 posições corretas (pares simétricos) são: (±3,0), (±2,±1), (±2,0), (±2,∓1), (±1,±2), (±1,±1), (±1,0), (±1,∓1), (±1,∓2), (0,±3), (0,±2), (0,±1).
7. **Transposição**: o VVenC não gira coordenadas, ele **permuta os coeficientes** com 4 tabelas fixas (ver `filterLumaSample`). A rotação usada no kernel não corresponde a nenhuma delas.
8. **Escala**: shift 7 (`m_NUM_BITS-1`) com arredondamento 64; o kernel usava 6.
9. **Clipping**: os valores são reais por coeficiente. Conjuntos fixos usam 1024 (= sem clip em 10 bits); APS usa {1024, 128, 32, 8}. O vetor do kernel tinha zeros, o que anulava taps.
10. **Classe 0 zerada**: o VVenC usa `m_classToFilterMapping[set][0]` também para a classe 0.
11. **VB no filtro**: substituição simétrica de linhas e shift +3 (`>>10`) nas duas linhas vizinhas da VB. Não existia.
12. **Contexto do codificador**: o VVenC filtra cada CTU com seu próprio conjunto (16 fixos + até 8 APS) ou não filtra; a entrada é a reconstrução pós-SAO (`m_tempBuf`), não o vídeo original.

Também: o mapa de classes era gravado por pixel (`uint32` W×H, 2 mapas ≈ 16 MB em 1080p). Aqui é 1 byte por bloco 4x4 (`classe | transpose<<5`). A thread de NVML do original nunca era iniciada.

## 2. Compilação

```bash
# CPU apenas (o --AlfGpu usa o backend de emulação: serve para validar bit-exatidão)
cmake -S . -B build/base -DCMAKE_BUILD_TYPE=Release && cmake --build build/base -j

# com CUDA (precisa de CUDA Toolkit + driver; NVML vem com o driver)
cmake -S . -B build/cuda -DCMAKE_BUILD_TYPE=Release -DVVENC_ENABLE_CUDA_ALF=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build/cuda -j
```
As duas builds gravam em `bin/release-static/`. Faça uma de cada vez ou copie os binários.

## 3. Uso

```bash
vvencFFapp -i seq.yuv -s 1920x1080 -fr 30 --InputBitDepth 8 --preset medium -t 0 -b out.266 -o rec.yuv \
           --AlfGpu 1 --AlfProf 1
```
* `--AlfGpu 0` → VVenC original (CPU, SIMD AVX2).
* `--AlfGpu 1` → a GPU faz **gradientes, classificação, índice de transformação geométrica, as estatísticas de covariância (luma e croma) e a filtragem** do luma. Classificação e covariâncias rodam no **início** do ALF e substituem as do VVenC: a derivação dos filtros recebe as matrizes calculadas pela GPU. O quadro e as classes ficam na memória da GPU até a filtragem.
* `--AlfGpu 2` → só o filtro na GPU, reutilizando a classificação que o VVenC já calculou para as estatísticas.
* `--AlfGpu 3` → modo verificação: roda GPU e CPU, compara amostra a amostra, classe a classe e cada valor `float` das covariâncias (bit a bit), mantém o resultado da CPU e imprime `VERIFY: ... BIT-EXACT`.
* `--AlfProf 1` → relatório de tempo e energia ao final (stderr).

Variáveis de ambiente: `VVENC_ALF_GPU=n` (equivale a `--AlfGpu` para o `vvencapp`), `VVENC_ALF_PROF=1`, `VVENC_ALF_GPU_EMU=1` (força emulação), `VVENC_ALF_GPU_DEVICE=n`, `VVENC_ALF_PROF_CSV=res.csv` (acrescenta uma linha por execução), `VVENC_ALF_PROF_LABEL=nome`, `VVENC_ALF_PROF_IDLE_MS=500` (janela da linha de base ociosa), `VVENC_ALF_DUMP=dir` e `VVENC_ALF_DUMP_MAX=8` (dump de quadros para o benchmark).

Pipeline da GPU: `VVENC_ALF_GPU_STREAMS=1..4` (streams CUDA por thread do codificador, padrão 2), `VVENC_ALF_GPU_CHUNK_ROWS=n` (linhas de CTU por pedaço do pipeline; padrão automático ≈ 4 pedaços por quadro), `VVENC_ALF_GPU_ROWMODE=1` (volta ao modo antigo, uma chamada por linha de CTUs, para comparação). Recomenda-se `CUDA_MODULE_LOADING=EAGER` para não pagar o carregamento preguiçoso dos kernels na primeira chamada.

### O que vai para a GPU
O ALF do VVenC tem 3 etapas por quadro: `ALF_GET_STATISTICS` (classificação + covariâncias, só no codificador), `ALF_DERIVE_FILTER` (Wiener/Cholesky, decisões por CTU, APS) e `ALF_RECONSTRUCT` (o filtro normativo). A GPU substitui o luma de `ALF_RECONSTRUCT`. Ficam na CPU: croma, CCALF e as CTUs cruzadas por fronteiras virtuais/slices/tiles com padding especial (raras com a configuração padrão).

**Fluxo do `--AlfGpu 1` por imagem (padrão):**

```
SAO de todas as CTUs ─► [GPU] classifyFrame: envia o quadro 1x, gradientes + classe + transposição de cada bloco 4x4,
                        devolve só as classes (1 byte/bloco)                       ← início do ALF (ALF_GET_STATISTICS)
                     ─► [GPU] covarianceFrame: matrizes E, vetores y e pixAcc de cada unidade de estatística
                        (ASU/CTU) e classe, luma (classes residentes) e croma; envia o original, devolve as matrizes
                     ─► [CPU] derivação Wiener, decisões on/off e conjunto de filtros por CTU, APS (ALF_DERIVE_FILTER)
                     ─► [GPU] filterFrameResident: sobe só o mapa de CTUs + APS, filtra com o quadro e as classes
                        que já estão na GPU (transformação geométrica = permutação dos coeficientes), devolve o quadro
                     ─► [CPU] croma, CCALF                                          ← fim do ALF
```
Para isso as tarefas de estatística esperam o SAO do quadro inteiro (`EncSlice.cpp`). Com IFP (`--IFP 1`) as decisões são tomadas linha a linha, então o codificador usa o modo linha: classificação na CPU para as estatísticas e GPU (classificação + filtro) por linha de CTUs. `VVENC_ALF_GPU_ROWMODE=1` força o modo linha. Ficam sempre na CPU as CTUs cruzadas por fronteiras virtuais/slices/tiles com padding especial (raras com a configuração padrão).

**Dentro da chamada (pipeline massivamente paralelo):** o quadro é dividido em pedaços de linhas de CTU; cada pedaço vai para um stream CUDA (`VVENC_ALF_GPU_STREAMS`), com eventos garantindo que a classificação de um pedaço só começa depois que as linhas de halo dos pedaços vizinhos chegaram:

```
stream 0:  [H2D p0][classif p0][filtro p0][D2H p0]         [H2D p2][classif p2]...
stream 1:          [H2D p1]   [classif p1][filtro p1][D2H p1]        [H2D p3]...
CPU:       empacota p0, p1, ...                    desempacota p0 assim que o D2H p0 termina
```
Cópia H2D, kernels, cópia D2H e o desempacotamento na CPU se sobrepõem. O relatório mostra a soma de cada etapa e a **linha do tempo sobreposta** (do primeiro H2D ao último D2H).

**Kernel de classificação (`kAlfClassify`).** Bloco CUDA de 256 threads cobre 128x32 amostras (32x8 blocos 4x4):
1. carga coalescida de 134x38 amostras (halo de 3) para shared memory;
2. cada célula 2x2 do laplaciano (os 4 gradientes, posições par/par e ímpar/ímpar) é calculada **uma única vez** (66x18 células, ~4,6 por thread) e guardada em shared memory;
3. uma thread por bloco 4x4 soma suas 4x4 células (3x4 na fronteira virtual) e calcula direção/atividade.

Blocos 4x4 vizinhos compartilham 3/4 das células da janela 8x8, então a fase 2 elimina ~3,5x da aritmética do kernel antigo (que recalculava a janela inteira por bloco) e deixa o trabalho uniforme por thread. Um quadro 1080p tem 15x34 = 510 blocos CUDA de classificação e 60x34 = 2040 de filtro: dezenas de milhares de threads, o suficiente para ocupar todos os SMs.

**Kernel de filtro (`kAlfFilterLuma`).** 32x8 threads, tile 32x32 amostras + halo 3 em shared memory, 4 amostras por thread; cada thread lê o conjunto de filtros da sua CTU (fixo ou APS) e a classe do seu bloco 4x4.

**Covariâncias bit-exatas.** O VVenC soma, para cada bloco 4x4, os 16 produtos de cada elemento em **inteiros** e só então converte para `float` e acumula na matriz da classe do bloco, bloco a bloco em ordem raster. Na GPU: as somas inteiras são calculadas em paralelo (cada bloco CUDA monta em shared memory os vetores `ELocal` de 16 blocos 4x4 de uma vez: vizinhos simétricos, permutação pela transposição, clipping não linear, linhas cortadas na fronteira virtual) e **uma thread por (unidade, elemento da matriz)** percorre os blocos na mesma ordem da CPU, acumulando em `float` com arredondamento IEEE (`__fadd_rn`), na mesma sequência de arredondamentos. Paralelismo: unidades × elementos (105 por unidade de luma com ALF linear, 1509 com não linear; 36/477 no croma). O modo 3 compara todos os valores bit a bit.

### Relatório (`--AlfProf 1`)
* Tempo por etapa (stats, derive, reconstruct, filtro luma CPU, chamada GPU, CCALF), % do tempo total do codificador e energia RAPL dentro de cada escopo. `luma classif. CPU` é a classificação 4x4 do VVenC e `luma classif. GPU call` a chamada `classifyFrame` (ambas dentro de `ALF stats`); `luma filter CPU` é o filtro 7x7 (dentro de `ALF reconstruct`). O `avg` dessas duas linhas é **por CTU**.
* Bloco **WHOLE ALF** (a medida para comparar o impacto): tudo o que o codificador faz para ALF + CCALF, do início das estatísticas ao fim da reconstrução — classificação (CPU ou GPU), covariâncias, derivação de Wiener, decisões por CTU, APS, filtragem luma e croma, CCALF. Mostra total, por imagem codificada e por imagem com ALF:
  * tempo = soma dos escopos do ALF, e tempo de parede com o ALF ativo (iguais com `-t 0`);
  * quanto disso é classificação luma, covariâncias, filtragem luma e o resto (que fica na CPU: derivação de Wiener, decisões, APS, filtro de croma, CCALF);
  * energia da CPU (RAPL dentro dos escopos do ALF, e acima da linha ociosa), energia da GPU (potência NVML amostrada a cada 2 ms, integrada **só enquanto o ALF está ativo**, e acima da ociosa), e CPU + GPU.
  * O CSV (`VVENC_ALF_PROF_CSV`) tem as colunas `whole_alf_ms`, `whole_alf_wall_ms`, `whole_alf_cpu_J`, `whole_alf_gpu_J`, `enc_pictures`, `alf_pictures`.
* GPU: empacotamento no host, **H2D**, **kernel de classificação**, **kernel de filtro**, **D2H**, desempacotamento, **comunicação** (H2D+D2H), **execução** (kernels), **linha do tempo sobreposta** (com o pipeline, menor que a soma das etapas) e **total** por chamada, com bytes e GB/s. No modo quadro há 1 chamada por quadro (que tem ALF de luma ligado).
* Energia: CPU (RAPL do pacote, linha de base ociosa, estimativa proporcional ao tempo do ALF) e GPU (contador de energia da NVML, amostragem de potência a 2 ms, pico, energia dinâmica = total − ociosa × tempo).

Para atribuir tempo/energia ao ALF sem ambiguidade, use `-t 0` (uma thread). Com várias threads os tempos por etapa são somas entre threads.
RAPL exige permissão: `sudo chmod o+r /sys/class/powercap/intel-rapl:*/energy_uj` (ou rodar como root).

## 4. Benchmark isolado (`alf_bench`)

```bash
cd test/alfgpu && make bench CUDA_ARCH=native
# quadros reais do codificador (recomendado)
VVENC_ALF_DUMP=/tmp/alfdump vvencFFapp ... --AlfGpu 0
./alf_bench --dump /tmp/alfdump/alf_poc00016.bin --min-seconds 3 --csvlog bench.csv --label BQTerrace_qp32
# ou o seu CSV antigo (todas as CTUs com o conjunto fixo --set)
./alf_bench --csv original_0.csv --width 1920 --height 1080 --ctu 128 --set 0 --out filtrada.csv --classes classes.csv
```
Mede, por quadro e com média/desvio/mínimo/mediana: CPU (classificação, filtro, total, energia RAPL) e GPU (H2D, classificação, filtro, D2H, comunicação, execução, total de parede, energia NVML com e sem a parcela ociosa). Opções: `--pageable` (sem memória pinned), `--cpu-class` (classificação vinda da CPU), `--iters`, `--warmup`. Ao final compara GPU × CPU e imprime `BIT-EXACT`; no modo `--dump`, também compara a classificação com a do VVenC.

Depois da medição com um stream, o `alf_bench` passa o mesmo quadro pelo **backend do codificador** (`AlfGpuCuda.cu`, exatamente a chamada do modo quadro, com pedaços e vários streams) e imprime a linha do tempo sobreposta e o tempo da chamada. Teste combinações:
```bash
for s in 1 2 4; do for c in 0 1 2 4; do VVENC_ALF_GPU_STREAMS=$s VVENC_ALF_GPU_CHUNK_ROWS=$c ./alf_bench --dump quadro.bin --label s${s}_c${c} --csvlog pipe.csv; done; done
```

A referência de CPU do `alf_bench` é o núcleo **escalar**. A comparação justa para o artigo é contra o VVenC com SIMD, ou seja, a linha `luma filter CPU` do `--AlfProf` com `--AlfGpu 0`.

## 5. Validação feita (sem GPU nesta máquina)

* `test_alf_core`: núcleo × `deriveClassificationBlk` e `filterBlk` do VVenC (escalar **e** AVX2), 5 resoluções, CTU 64/128, 5 conteúdos sintéticos (incl. ruído branco extremo), clipping linear e não linear: 3,4 M blocos e 54,8 M amostras, **0 diferenças**. Um teste de mutação (VB quebrada) é detectado.
* `run_bitexact_tests.sh`: 9 configurações (RA/AI, CTU 64/128, ASU 128 com CTU 64, ALF não linear em luma e em luma+croma, preset medium, IFP com 4 threads, tiles, 1080p com 2 threads, altura 472 não múltipla da CTU). `--AlfGpu 1/2/3` geram **bitstream e reconstrução idênticos** ao `--AlfGpu 0` em três variantes: modo quadro (padrão), modo linha (`VVENC_ALF_GPU_ROWMODE=1`) e pedaços de 1 linha de CTU (`VVENC_ALF_GPU_CHUNK_ROWS=1`). O modo 3 reporta BIT-EXACT em todas, incluindo a comparação bit a bit de até 619 mil valores de covariância por execução (com IFP as covariâncias ficam na CPU).
* O backend de emulação executa as **mesmas funções de fase** dos kernels, bloco CUDA por bloco e thread por thread, com os mesmos pedaços do pipeline; por isso o tiling, os halos e os índices de shared memory dos kernels novos estão validados mesmo sem GPU.
* `alf_bench --dump` (núcleo da CPU em um quadro 1080p inteiro) = luma reconstruído pelo VVenC em todas as 2.073.600 amostras.
* `AlfGpuCuda.cu` e `alf_bench.cu` compilam com nvcc 13.4 (`sm_120`, RTX 50xx: classificação 76 registradores / 29 KB de shared memory, filtro 48 registradores / 2,9 KB, sem spills); a build CMake com `-DVVENC_ENABLE_CUDA_ALF=ON` compila e linka. **Os kernels não foram executados em GPU aqui**: rode primeiro `--AlfGpu 3` e `alf_bench` na sua máquina.
