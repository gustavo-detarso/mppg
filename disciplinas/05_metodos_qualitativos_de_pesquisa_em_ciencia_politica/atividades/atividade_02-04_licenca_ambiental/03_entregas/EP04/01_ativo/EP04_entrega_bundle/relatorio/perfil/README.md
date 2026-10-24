# Perfil FGV/ABNT canônico

`fgv_abnt.toml` é a fonte única de verdade para a apresentação institucional deste bundle.

O fluxo é:

```text
fgv_abnt.toml
   ├──> aplicar_fgv_abnt.py ──> DOCX final
   ├──> gerar_fgv_abnt_sty.py ──> fgv_abnt_generated.sty
   └──> compilar_org_pdf.sh ──> DOCX + PDF + logs/hashes
```

## Regra de governança

- altere margens, fontes, cores, recuos, cabeçalho, rodapé, dimensões de figuras e demais parâmetros **somente em `fgv_abnt.toml`**;
- não edite `fgv_abnt_generated.sty`: ele é regenerado automaticamente;
- `modelo/FGV_ABNT_EP04_reference.docx` permanece como base estrutural do Pandoc, mas os parâmetros normativos de layout são sobrescritos pelo perfil TOML;
- o DOCX final registra em suas propriedades o SHA-256 do perfil efetivamente aplicado;
- o log registra os hashes do perfil, do `.sty`, do DOCX e do PDF.

Assim, o backend DOCX/PDF e um eventual backend LaTeX derivam da mesma configuração.
