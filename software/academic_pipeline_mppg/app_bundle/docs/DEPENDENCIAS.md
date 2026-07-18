# Dependências da rc10.1 estável

## Python

Instale as dependências Python com:

```bash
pip install -r requirements.txt
```

Pacotes principais:

- openai
- pydantic
- python-dotenv
- pypdf
- python-docx
- openpyxl

## PDF

Para exportar PDF:

- Emacs com Org-mode;
- LuaLaTeX, XeLaTeX ou PDFLaTeX;
- Biber;
- biblatex;
- biblatex-apa para APA;
- biblatex-abnt para ABNT;
- arquivo local `app_bundle/misc/academic-writing.el`;
- arquivos `.sty` em `app_bundle/misc/fgv/`.

## DOCX

O DOCX básico usa `python-docx`.

Para DOCX com APA/ABNT rigoroso por CSL, instale também:

- Pandoc;
- arquivo CSL em `app_bundle/templates/csl/`.

## Mapa mental

Para renderizar mapa mental PlantUML:

- Java;
- PlantUML por comando `plantuml` ou caminho do `.jar` no TOML.
