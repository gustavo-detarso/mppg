#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, tomllib


def esc_tex(s: str) -> str:
    return (s.replace('\\', r'\textbackslash{}')
             .replace('&', r'\&')
             .replace('%', r'\%')
             .replace('$', r'\$')
             .replace('#', r'\#')
             .replace('_', r'\_')
             .replace('{', r'\{')
             .replace('}', r'\}')
             .replace('~', r'\textasciitilde{}')
             .replace('^', r'\textasciicircum{}'))


def main():
    ap = argparse.ArgumentParser(description='Gera um .sty LaTeX a partir do perfil FGV/ABNT canônico.')
    ap.add_argument('profile', type=Path)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()

    with args.profile.open('rb') as f:
        p = tomllib.load(f)

    page = p['pagina']; typo = p['tipografia']; colors = p['cores']; doc = p['documento']; titles = p['titulos']
    sha = hashlib.sha256(args.profile.read_bytes()).hexdigest()

    text = rf'''% GERADO AUTOMATICAMENTE. NÃO EDITAR.
% Fonte canônica: {args.profile.name}
% SHA-256 do perfil: {sha}
\NeedsTeXFormat{{LaTeX2e}}
\ProvidesPackage{{fgv-abnt-generated}}[2026/09/20 Perfil FGV/ABNT gerado]
\RequirePackage{{geometry}}
\geometry{{a4paper,top={page['margem_superior_cm']}cm,left={page['margem_esquerda_cm']}cm,right={page['margem_direita_cm']}cm,bottom={page['margem_inferior_cm']}cm}}
\RequirePackage{{iftex}}
\ifPDFTeX
  % pdfLaTeX não usa fontspec; mantenha uma fonte Times compatível.
  \RequirePackage{{newtxtext,newtxmath}}
\else
  \RequirePackage{{fontspec}}
  \setmainfont{{{esc_tex(typo['fonte_principal'])}}}
\fi
\RequirePackage{{xcolor}}
\definecolor{{FGVTitle}}{{HTML}}{{{colors['titulo']}}}
\definecolor{{FGVHeadingOne}}{{HTML}}{{{colors['heading1']}}}
\definecolor{{FGVHeadingTwo}}{{HTML}}{{{colors['heading2']}}}
\definecolor{{FGVHeadingThree}}{{HTML}}{{{colors['heading3']}}}
\definecolor{{FGVRule}}{{HTML}}{{{colors['linha_titulo']}}}
\definecolor{{FGVHeaderGray}}{{HTML}}{{{colors['cabecalho']}}}
\RequirePackage{{fancyhdr}}
\setlength{{\headheight}}{{18pt}}
\setlength{{\headsep}}{{{page['distancia_cabecalho_cm']}cm}}
\setlength{{\footskip}}{{{page['distancia_rodape_cm'] + 0.5}cm}}
\newcommand{{\FGVHeaderText}}{{{esc_tex(doc['cabecalho_texto'])}}}
\newcommand{{\FGVFooterPrefix}}{{{esc_tex(doc['rodape_prefixo'])}}}
\fancypagestyle{{fgvabnt}}{{%
  \fancyhf{{}}%
  \fancyhead[C]{{\color{{FGVHeaderGray}}\fontsize{{{typo['cabecalho_pt']}}}{{9}}\selectfont \FGVHeaderText}}%
  \fancyfoot[C]{{\fontsize{{{typo['rodape_pt']}}}{{10}}\selectfont \FGVFooterPrefix\thepage}}%
  \renewcommand{{\headrulewidth}}{{0pt}}%
}}
\pagestyle{{fgvabnt}}
\RequirePackage{{titlesec}}
\titleformat{{\section}}{{\bfseries\color{{FGVHeadingOne}}\fontsize{{{titles['heading1_pt']}}}{{14}}\selectfont}}{{\thesection}}{{0.5em}}{{}}
\titleformat{{\subsection}}{{\bfseries\color{{FGVHeadingTwo}}\fontsize{{{titles['heading2_pt']}}}{{13}}\selectfont}}{{\thesubsection}}{{0.5em}}{{}}
\titleformat{{\subsubsection}}{{\bfseries\color{{FGVHeadingThree}}\fontsize{{{titles['heading3_pt']}}}{{12}}\selectfont}}{{\thesubsubsection}}{{0.5em}}{{}}
\setlength{{\parindent}}{{{typo['corpo_recuo_primeira_linha_cm']}cm}}
\setlength{{\parskip}}{{0pt}}
'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding='utf-8')
    print(f'PASS: STY gerado: {args.output}')
    print(f'PASS: perfil SHA-256: {sha}')

if __name__ == '__main__':
    main()
