# AP-005E.1 — Estratégia de instalação e entrypoints

## Objetivo

Congelar a superfície instalável atualmente declarada antes de construir artefatos e testar uma instalação isolada.

## Superfícies preservadas

- distribuição `academic-pipeline-mppg`, versão `0.1.0`;
- console script `academic-pipeline`;
- execução `python -m academic_pipeline`;
- função pública `academic_pipeline.main` e `__all__ = ["main"]`;
- bridge `academic_pipeline.legacy` para o runtime histórico;
- script histórico `academic_pipeline_rc10.py` como compatibilidade.

## Exclusões

- não renomear pacotes ou módulos;
- não remover wrappers, facades ou aliases;
- não alterar o runtime acadêmico;
- não preencher `project.dependencies` nesta subfase;
- não decidir package data por contagem bruta;
- não antecipar reorganizações da AP-006.

## Interpretação das evidências

A instalação observada no Pipenv atual prova somente que existe uma distribuição `0.1.0` importável a partir de `site-packages` e um console script no `bin` desse ambiente. Ela não é gate de encerramento porque pode refletir instalação anterior e não foi construída em descendente temporário limpo.

Os 274 arquivos rastreados sob `academic_pipeline` e `app_bundle` não podem ser comparados diretamente aos 141 registros da distribuição instalada. Os universos têm semânticas diferentes.

## Gates da AP-005E.2

1. build wheel and sdist from a clean temporary descendant.
2. inspect exact archive manifests and reject accidental residues.
3. install wheel into a fresh temporary virtual environment.
4. remove PYTHONPATH and run from outside the checkout.
5. prepend the temporary environment bin directory to PATH.
6. prove academic_pipeline.__file__ belongs to the temporary environment.
7. prove academic-pipeline resolves to the temporary environment.
8. compare academic-pipeline and python -m academic_pipeline help.
9. exercise the legacy bridge without importing from the checkout.
10. characterize package data required by operational commands.
11. characterize hardcoded and sibling-helper layout risks.

## Critério de aplicação

A AP-005E.3 somente poderá alterar metadata, package data ou entrypoints quando a AP-005E.2 reproduzir um defeito concreto no artefato instalado. Caso todos os gates passem, a aplicação será formalmente `no-op`.
