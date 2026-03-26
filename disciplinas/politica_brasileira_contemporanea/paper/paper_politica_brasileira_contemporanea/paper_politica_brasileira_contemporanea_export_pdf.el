(require 'package)
(package-initialize)
(require 'org)
(require 'ox)
(require 'ox-latex)
(require 'oc)
(require 'oc-biblatex)
(setq org-export-use-babel nil)
(setq org-confirm-babel-evaluate nil)
(setq org-latex-pdf-process
      '("lualatex -interaction=nonstopmode -file-line-error %f"
        "biber %b"
        "lualatex -interaction=nonstopmode -file-line-error %f"
        "lualatex -interaction=nonstopmode -file-line-error %f"))
(find-file "/home/gustavodetarso/Documentos/mppg/disciplinas/politica_brasileira_contemporanea/paper/paper_politica_brasileira_contemporanea/paper_politica_brasileira_contemporanea.org")
(org-latex-export-to-pdf)
