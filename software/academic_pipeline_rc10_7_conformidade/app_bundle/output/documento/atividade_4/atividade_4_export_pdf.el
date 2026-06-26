
(require 'org)
(require 'ox-latex)
(require 'oc)
(ignore-errors (require 'oc-biblatex))
(load-file "/home/gustavodetarso/Documentos/mppg/software/academic_pipeline_rc10_7_conformidade/app_bundle/misc/academic-writing.el")
;; Evita erro do Org 9.5 com forma antiga/dotted de org-cite-export-processors.
;; O pipeline usa citações LaTeX diretas (\parencite/\textcite) e injeta BibLaTeX via LATEX_HEADER.
(setq org-cite-export-processors '((latex biblatex) (t basic)))
(setq org-confirm-babel-evaluate nil)
(setq org-latex-pdf-process '("lualatex -interaction nonstopmode -shell-escape -output-directory %o %f" "biber %b" "lualatex -interaction nonstopmode -shell-escape -output-directory %o %f" "lualatex -interaction nonstopmode -shell-escape -output-directory %o %f"))
(find-file "/home/gustavodetarso/Documentos/mppg/software/academic_pipeline_rc10_7_conformidade/app_bundle/output/documento/atividade_4/atividade_4.org")
(org-latex-export-to-pdf)
