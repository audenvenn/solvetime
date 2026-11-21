| model                          |       size |     params | backend    | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | --------------: | -------------------: |
| inkling2_loop ?B Q3_K - Medium |  12.68 GiB |    26.07 B | MTL,BLAS   |       2 |   1 |           pp512 |         28.57 ± 0.08 |
| inkling2_loop ?B Q3_K - Medium |  12.68 GiB |    26.07 B | MTL,BLAS   |       2 |   1 |           tg128 |         15.52 ± 0.13 |

build: 29b52d398 (12600)
| model                          |       size |     params | backend    | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | --------------: | -------------------: |
| inkling2_loop ?B IQ3_XXS - 3.0625 bpw |  10.93 GiB |    26.07 B | MTL,BLAS   |       2 |   1 |           pp512 |        319.15 ± 1.34 |
| inkling2_loop ?B IQ3_XXS - 3.0625 bpw |  10.93 GiB |    26.07 B | MTL,BLAS   |       2 |   1 |           tg128 |         33.62 ± 0.04 |

build: 29b52d398 (12600)
