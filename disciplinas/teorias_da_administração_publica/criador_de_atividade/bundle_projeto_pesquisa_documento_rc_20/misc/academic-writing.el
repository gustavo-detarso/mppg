;;; academic-writing.el --- Escrita acadêmica com Org Cite + classes FGV -*- lexical-binding: t; -*-

(require 'org)
(require 'ox)
(require 'ox-latex)
(require 'oc)
(require 'oc-biblatex)
(require 'cl-lib)

(defgroup academic-writing nil
  "Configurações para escrita acadêmica com Org e LaTeX."
  :group 'org
  :prefix "gm/academic-")

(defcustom gm/academic-latex-compiler "lualatex"
  "Compilador LaTeX padrão para exportações acadêmicas."
  :type 'string
  :group 'academic-writing)

(defcustom gm/academic-default-class "fgv-paper"
  "Classe LaTeX padrão sugerida."
  :type 'string
  :group 'academic-writing)

(defcustom gm/academic-default-bibfile "referencias.bib"
  "Arquivo .bib padrão sugerido."
  :type 'string
  :group 'academic-writing)

(defun gm/academic-configure-latex-export ()
  "Configura o pipeline LaTeX para escrita acadêmica."
  (setq org-latex-compiler gm/academic-latex-compiler)
  (setq org-latex-pdf-process
        '("lualatex -interaction=nonstopmode -file-line-error %f"
          "biber %b"
          "lualatex -interaction=nonstopmode -file-line-error %f"
          "lualatex -interaction=nonstopmode -file-line-error %f")))

(defun gm/academic-configure-citations ()
  "Configura Org Cite para exportação LaTeX com BibLaTeX."
  (setq org-cite-insert-processor 'basic
        org-cite-follow-processor 'basic
        org-cite-activate-processor 'basic
        org-cite-export-processors '((latex . biblatex)
                                     (t . basic))))

(defun gm/academic--remove-class (name)
  "Remove NAME de `org-latex-classes` se existir."
  (setq org-latex-classes
        (cl-remove-if (lambda (it) (equal (car it) name))
                      org-latex-classes)))

(defun gm/academic-register-classes ()
  "Registra classes LaTeX para paper e dissertação FGV.

Regra arquitetural:
- paper -> fgv-paper
- dissertacao -> fgv-dissertacao"
  (gm/academic--remove-class "fgv-paper")
  (gm/academic--remove-class "fgv-dissertacao")

  (add-to-list
   'org-latex-classes
   '("fgv-paper"
     "\\documentclass[12pt,a4paper]{article}
[NO-DEFAULT-PACKAGES]
[PACKAGES]
\\usepackage{fgv-paper}
[EXTRA]"
     ("\\section{%s}" . "\\section*{%s}")
     ("\\subsection{%s}" . "\\subsection*{%s}")
     ("\\subsubsection{%s}" . "\\subsubsection*{%s}")
     ("\\paragraph{%s}" . "\\paragraph*{%s}")
     ("\\subparagraph{%s}" . "\\subparagraph*{%s}")))

  (add-to-list
   'org-latex-classes
   '("fgv-dissertacao"
     "\\documentclass[12pt,a4paper]{article}
[NO-DEFAULT-PACKAGES]
[PACKAGES]
\\usepackage{fgv-dissertacao}
[EXTRA]"
     ("\\section{%s}" . "\\section*{%s}")
     ("\\subsection{%s}" . "\\subsection*{%s}")
     ("\\subsubsection{%s}" . "\\subsubsection*{%s}")
     ("\\paragraph{%s}" . "\\paragraph*{%s}")
     ("\\subparagraph{%s}" . "\\subparagraph*{%s}"))))

(defun gm/academic-setup ()
  "Executa a configuração completa do módulo."
  (interactive)
  (require 'oc)
  (require 'oc-biblatex)
  (gm/academic-configure-latex-export)
  (gm/academic-configure-citations)
  (gm/academic-register-classes))

(defun gm/academic-check-environment ()
  "Mostra um diagnóstico rápido do ambiente acadêmico."
  (interactive)
  (message "Org=%s | oc-biblatex=%s | classes=(fgv-paper, fgv-dissertacao)"
           (org-version)
           (if (featurep 'oc-biblatex) "ok" "NÃO CARREGADO")))

(gm/academic-setup)

(provide 'academic-writing)
;;; academic-writing.el ends here
