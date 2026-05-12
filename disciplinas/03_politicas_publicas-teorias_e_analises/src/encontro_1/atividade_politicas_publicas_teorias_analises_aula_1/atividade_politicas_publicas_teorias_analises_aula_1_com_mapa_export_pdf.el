(load-file "/home/gustavodetarso/Documentos/mppg/disciplinas/teorias_da_administração_publica/criador_de_atividade/bundle_projeto_pesquisa_documento_rc_20/misc/academic-writing.el")
(require 'org)
    (require 'ox)
    (require 'ox-latex)
    (ignore-errors (require 'oc))
(ignore-errors (require 'oc-biblatex))
(find-file "/home/gustavodetarso/Documentos/mppg/disciplinas/teorias_da_administração_publica/criador_de_atividade/bundle_projeto_pesquisa_documento_rc_20/output/documento/atividade_politicas_publicas_teorias_analises_aula_1/atividade_politicas_publicas_teorias_analises_aula_1_com_mapa.org")
    (setq-local org-export-use-babel nil)
    (setq-local org-confirm-babel-evaluate nil)
    (setq org-cite-insert-processor 'basic
      org-cite-follow-processor 'basic
      org-cite-activate-processor 'basic
      org-cite-export-processors '((latex biblatex) (t basic)))
(setq-local org-cite-export-processors '((latex biblatex) (t basic)))
(org-latex-export-to-latex)
