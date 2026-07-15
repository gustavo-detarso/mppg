
        # AP-004 — convenção canônica de nomes internos (v4.2)

        ## Finalidade

        Esta convenção disciplina a normalização de nomes sem alterar comportamento,
        superfícies de entrada, documentos gerados ou caminhos operacionais. O
        inventário v4.2 distingue marcadores estruturais inequívocos de palavras
        contextuais que podem representar conceitos legítimos do domínio.

        ## Gatilhos acionáveis automáticos

        Somente os seguintes marcadores podem criar automaticamente um candidato:

        - `rcNN` e equivalentes de release candidate;
        - `vNN`, `v1_18`, `v0_3_1` e equivalentes;
        - palavras explícitas de versão seguidas por número;
        - prefixos de refatoração `apNNN`, como `_ap003d_` e `_ap003f_`.

        Palavras como `final`, `novo`, `new`, `old`, `legacy`, `original` e `pre`
        são apenas evidência contextual. Elas não criam candidato por si mesmas.
        `copy` e `backup` são verbos legítimos e não são marcadores de nomenclatura.

        ## Regras canônicas

        1. **Módulos e arquivos Python produtivos** usam `snake_case` e nomes
           semânticos. A sugestão deve descrever responsabilidade real.
        2. **Funções e métodos** usam `snake_case` e preservam verbos de ação.
        3. **Métodos especiais `__dunder__`** ficam integralmente fora da análise.
        4. **Classes** usam `PascalCase`; classes privadas podem usar `_PascalCase`.
        5. **Constantes** usam `UPPER_SNAKE_CASE`.
        6. **`legacy`/`legado`** permanece quando identifica uma camada real de
           compatibilidade; sua necessidade será revista na AP-004E.
        7. **Aliases numéricos**, como `stage_001` e `dispatch_001`, são opacos e
           exigem nome semântico explícito.
        8. **Imports** são consumidores e não duplicam a decisão do módulo ou símbolo
           de origem.
        9. **Entrypoints** são superfícies relacionadas ao candidato principal do
           módulo; não aparecem como candidatos independentes.
        10. **Testes e documentação** registram consumidores, contratos e história.
        11. **Saídas operacionais, execuções anteriores, instaladores, assets e
            scripts históricos de aplicação/atualização/migração na raiz** ficam fora
            da AP-004.
        12. **Colisões de destino** suspendem a sugestão e elevam os envolvidos para
            alto risco.
        13. Sugestões que ainda contenham `original`, `pre`, `stage_N`, `dispatch_N`
            ou cauda numérica opaca são rejeitadas.

        ## Estruturas do inventário

        - `raw_occurrences`: levantamento amplo de ocorrências e evidências;
        - `actionable_candidates`: decisões possíveis sobre módulos e símbolos;
        - `contextual_review_occurrences`: palavras contextuais não acionáveis;
        - `protected_operational_names`: caminhos operacionais preservados;
        - `historical_references`: testes e documentação como evidência;
        - `destination_collisions`: destinos propostos por mais de uma origem;
        - `manual_review_required`: candidatos de alto risco.

        ## Critérios de classificação

        ### Renomeação segura

        Símbolo privado/local ou arquivo Python produtivo interno, com marcador
        estrutural inequívoco, sugestão semântica conservadora e sem consumidor
        externo, string dinâmica ou colisão.

        ### Renomeação com compatibilidade

        Módulo ou símbolo público, exportado, consumido externamente ou relacionado
        a entrypoint, cuja migração exige wrapper ou alias transitório documentado.

        ### Renomeação de alto risco

        Marcador estrutural presente, mas sem sugestão semântica segura, com alias
        opaco, referência dinâmica, colisão ou múltiplos consumidores não resolvidos.

        ### Nome que deve permanecer

        Diretório físico reservado à AP-006, xfail congelado, histórico auditável,
        script operacional protegido ou camada real de compatibilidade `legacy`.

        ## Limites

        O diretório `academic_pipeline_rc10_7_conformidade`, os três xfails, a
        semântica da CLI, instaladores, assets, caminhos de implantação e conteúdo
        documental gerado permanecem inalterados.

        Nenhuma renomeação será feita por substituição textual global. Aplicadores
        futuros deverão validar `HEAD`, hashes, AST, conjunto permitido de arquivos,
        escrita atômica e rollback integral.

