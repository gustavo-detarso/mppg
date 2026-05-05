;;; academic-writing.el --- Escrita acadêmica com Org Cite + BibLaTeX -*- lexical-binding: t; -*-

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
  "Classe LaTeX padrão sugerida para papers acadêmicos."
  :type 'string
  :group 'academic-writing)

(defcustom gm/academic-default-bibfile "referencias_paper_apa.bib"
  "Arquivo .bib padrão sugerido nos templates acadêmicos."
  :type 'string
  :group 'academic-writing)

(defun gm/academic-configure-latex-export ()
  "Configura o pipeline LaTeX para escrita acadêmica.
Usa exatamente a sequência que já funcionou no ambiente do usuário."
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

(defun gm/academic-register-classes ()
  "Registra classes LaTeX úteis para escrita acadêmica.

IMPORTANTE: os arquivos fgv-paper e fgv-dissertacao disponíveis no projeto
são pacotes .sty, não classes .cls. Portanto, as classes Org registradas aqui
usam classes LaTeX padrão — article/report — combinadas com \\usepackage{...}.
Isso evita os erros 'File `fgv-paper.cls` not found' e
'File `fgv-dissertacao.cls` not found' no Emacs batch.

Também é importante que \\usepackage{fgv-paper} ou \\usepackage{fgv-dissertacao}
venha antes de [EXTRA], porque as linhas #+LATEX_HEADER: entram em [EXTRA].
Se o pacote vier depois, macros como \\usepapercover, \\institution, \\autor,
\\titulo etc. ficam indefinidas."
  (setq org-latex-classes
        (cl-remove-if (lambda (it)
                        (member (car it) '("fgv-paper" "fgv-dissertacao")))
                      org-latex-classes))

  ;; Paper: headings "*" viram \section. O arquivo real é fgv-paper.sty.
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
     ("\\paragraph{%s}" . "\\paragraph*{%s}")))

  ;; Dissertação: headings "*" viram \chapter. O arquivo real é fgv-dissertacao.sty.
  (add-to-list
   'org-latex-classes
   '("fgv-dissertacao"
     "\\documentclass[12pt,a4paper,oneside]{report}
[NO-DEFAULT-PACKAGES]
[PACKAGES]
\\usepackage{fgv-dissertacao}
[EXTRA]"
     ("\\chapter{%s}" . "\\chapter*{%s}")
     ("\\section{%s}" . "\\section*{%s}")
     ("\\subsection{%s}" . "\\subsection*{%s}")
     ("\\subsubsection{%s}" . "\\subsubsection*{%s}")
     ("\\paragraph{%s}" . "\\paragraph*{%s}"))))

(defun gm/academic-setup ()
  "Executa a configuração completa do módulo academic-writing."
  (interactive)
  (require 'oc)
  (require 'oc-biblatex)
  (gm/academic-configure-latex-export)
  (gm/academic-configure-citations)
  (gm/academic-register-classes))

(defun gm/academic-insert-textcite (key)
  "Insere citação narrativa nativa do Org para KEY."
  (interactive "sChave BibLaTeX/BibTeX: ")
  (insert (format "[cite/t:@%s]" key)))

(defun gm/academic-insert-parencite (key)
  "Insere citação parentética nativa do Org para KEY."
  (interactive "sChave BibLaTeX/BibTeX: ")
  (insert (format "[cite:@%s]" key)))

(defun gm/academic-insert-bibliography-footer ()
  "Insere o bloco nativo de bibliografia do Org Cite."
  (interactive)
  (insert "#+print_bibliography:\n"))

(defun gm/insert-fgv-paper-template ()
  "Insere um template base de paper em Org com Org Cite + BibLaTeX."
  (interactive)
  (insert
   (mapconcat
    #'identity
    (list
     "#+title: Título do paper"
     "#+author: Seu Nome"
     "#+date: \\today"
     "#+language: pt_BR"
     "#+options: toc:nil num:t title:nil html-postamble:nil ^:{}"
     "#+startup: indent"
     "#+latex_compiler: lualatex"
     (format "#+latex_class: %s" gm/academic-default-class)
     "#+latex_class_options: [12pt,a4paper]"
     "#+cite_export: biblatex backend=biber,style=apa,sortcites=true,sorting=nyt,giveninits=true,maxcitenames=2,maxbibnames=20,uniquelist=minyear"
     (format "#+bibliography: %s" gm/academic-default-bibfile)
     "#+latex_header: \\usepapercover"
     "#+latex_header: \\institution{Fundação Getulio Vargas}"
     "#+latex_header: \\programname{Mestrado Profissional em Políticas Públicas e Governo}"
     "#+latex_header: \\coursename{Política Brasileira Contemporânea}"
     "#+latex_header: \\disciplinename{Paper final da disciplina}"
     "#+latex_header: \\professorname{Professor da disciplina}"
     "#+latex_header: \\cityname{Brasília}"
     "#+latex_header: \\papertype{Paper apresentado como requisito de avaliação da disciplina}"
     "#+latex_header: \\covernote{Versão em Org-mode com org-cite e referências em APA via BibLaTeX.}"
     ""
     "#+latex: \\makemytitle"
     ""
     "#+begin_abstract"
     "Resumo do paper."
     "#+end_abstract"
     ""
     "#+latex: \\palavraschave{palavra-chave 1; palavra-chave 2; palavra-chave 3}"
     ""
     "* Introdução"
     ""
     "Texto introdutório com citação narrativa [cite/t:@palermo2000] e citação parentética [cite:@limongi2017]."
     ""
     "* Desenvolvimento"
     ""
     "Texto do desenvolvimento."
     ""
     "* Conclusão"
     ""
     "Texto conclusivo."
     ""
     "#+print_bibliography:")
    "\n")))

(defun gm/academic-check-environment ()
  "Mostra um diagnóstico rápido do ambiente de exportação acadêmica."
  (interactive)
  (message
   "Org=%s | oc-biblatex=%s | class=%s | bib=%s"
   (org-version)
   (if (featurep 'oc-biblatex) "ok" "NÃO CARREGADO")
   gm/academic-default-class
   gm/academic-default-bibfile))

(gm/academic-setup)

(provide 'academic-writing)
;;; academic-writing.el ends here
