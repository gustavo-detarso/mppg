(load-file "/home/gustavodetarso/Documentos/mppg/software/academic_pipeline/app_bundle/misc/academic-writing.el")
(require 'org)
    (require 'ox)
    (require 'ox-latex)
    (ignore-errors (require 'oc))
(ignore-errors (require 'oc-biblatex))
(find-file "/home/gustavodetarso/Documentos/mppg/software/academic_pipeline/app_bundle/output/documento/paper_politica_brasileira_contemporanea/paper_politica_brasileira_contemporanea.org")
    (setq-local org-export-use-babel nil)
    (setq-local org-confirm-babel-evaluate nil)
    (setq org-cite-insert-processor 'basic
      org-cite-follow-processor 'basic
      org-cite-activate-processor 'basic
      org-cite-export-processors '((latex biblatex) (t basic)))
(setq-local org-cite-export-processors '((latex biblatex) (t basic)))
(org-latex-export-to-latex)
