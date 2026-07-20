# AP-006D.4D — Contrato de preservação de `source_csv`

## Decisão

O campo `source_csv` deve permanecer literal nos quatro CSVs autoritativos e nas quatro cópias de cache. Seu valor representa a proveniência histórica do arquivo de triagem que originou os registros; ele não é dereferenciado em runtime pelo consumidor produtivo.

Nenhum dos oito CSVs e nenhum módulo produtivo foram alterados nesta materialização.

## Evidência

- Pares fonte-cache: **4**
- Linhas preservadas: **308**
- Valores únicos de `source_csv`: **1**
- Sinks de leitura derivados do campo: **0**
- Sinks de existência derivados do campo: **0**
- Sinks de serialização derivados: **4**
- Semântica: `historical_provenance_serialized_without_runtime_dereference`
- Fingerprint contratual: `6ef99836e4b7019e4dfb7921c5cc390ff7dcd979003908d251809a21185d3ebf`

## Política

É proibido reescrever os 308 valores, introduzir resolvedor de caminho ou alterar o consumidor com base apenas na aparência absoluta do campo. Qualquer mudança futura exige evidência nova de dereferência operacional. A regeneração dos caches continua regida pelo contrato da AP-006D.4C.

O manifesto verificável está em `docs/refactor/academic-pipeline/AP-006/ap006d4d_source_csv_provenance_contract.json`.
