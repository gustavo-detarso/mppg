(async () => {
  const NOME_ARQUIVO = "documento.pdf";

  function log(passo, msg, extra = "") {
    console.log(`%c[${passo}] ${msg}`, "color:#0b6;font-weight:bold", extra);
  }

  function warn(passo, msg, extra = "") {
    console.warn(`[${passo}] ${msg}`, extra);
  }

  function erro(passo, msg, extra = "") {
    console.error(`[${passo}] ${msg}`, extra);
  }

  function assinatura(bytes) {
    try {
      return new TextDecoder().decode(bytes.slice(0, 8));
    } catch {
      return "";
    }
  }

  function parecePdf(bytes) {
    return bytes && bytes.byteLength > 5 && assinatura(bytes).startsWith("%PDF-");
  }

  function baixarBytes(bytes, nomeArquivo = NOME_ARQUIVO) {
    const blob = new Blob([bytes], { type: "application/pdf" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = nomeArquivo;
    document.body.appendChild(a);
    a.click();
    a.remove();

    setTimeout(() => URL.revokeObjectURL(url), 10000);
  }

  async function tentarObjetoPdf(path, obj) {
    if (!obj || typeof obj.getData !== "function") {
      return false;
    }

    try {
      log("TESTE", `Tentando extrair via ${path}`);

      const data = await obj.getData();
      const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);

      console.table({
        caminho: path,
        paginas: obj.numPages || null,
        bytes: bytes.byteLength,
        assinatura: assinatura(bytes)
      });

      if (parecePdf(bytes)) {
        log("SUCESSO", `PDF extraído da memória: ${bytes.byteLength} bytes`);
        baixarBytes(bytes);
        return true;
      }

      warn("FALHA", `Objeto encontrado, mas não parece PDF válido: ${path}`);
      return false;
    } catch (e) {
      warn("ERRO", `Falhou ao tentar ${path}`, e);
      return false;
    }
  }

  const candidatos = [];
  const vistos = new WeakSet();

  function adicionar(path, obj) {
    if (!obj) return;
    if (typeof obj !== "object" && typeof obj !== "function") return;
    if (vistos.has(obj)) return;

    vistos.add(obj);

    if (typeof obj.getData === "function") {
      candidatos.push({ path, obj });
    }
  }

  log("1", "Procurando PDF em PDF_INSTANCES...");

  for (const [id, inst] of Object.entries(window.PDF_INSTANCES || {})) {
    adicionar(`PDF_INSTANCES.${id}.linkService.pdfDocument`, inst?.linkService?.pdfDocument);
    adicionar(`PDF_INSTANCES.${id}.pdfViewer.pdfDocument`, inst?.pdfViewer?.pdfDocument);
    adicionar(`PDF_INSTANCES.${id}.pdfDocument`, inst?.pdfDocument);
    adicionar(`PDF_INSTANCES.${id}._pdfDocument`, inst?._pdfDocument);
    adicionar(`PDF_INSTANCES.${id}.viewer.pdfDocument`, inst?.viewer?.pdfDocument);
  }

  log("2", "Procurando PDF em PDFViewerApplication...");

  adicionar("PDFViewerApplication.pdfDocument", window.PDFViewerApplication?.pdfDocument);
  adicionar("PDFViewerApplication.pdfViewer.pdfDocument", window.PDFViewerApplication?.pdfViewer?.pdfDocument);
  adicionar("PDFViewerApplication.pdfLinkService.pdfDocument", window.PDFViewerApplication?.pdfLinkService?.pdfDocument);

  log("3", "Fazendo varredura controlada em objetos conhecidos...");

  function varrer(obj, path, depth = 0) {
    if (!obj) return;
    if (depth > 6) return;
    if (typeof obj !== "object" && typeof obj !== "function") return;

    adicionar(path, obj);

    let keys = [];
    try {
      keys = Object.getOwnPropertyNames(obj);
    } catch {
      return;
    }

    for (const key of keys) {
      if (
        [
          "window",
          "document",
          "parent",
          "top",
          "frames",
          "self",
          "localStorage",
          "sessionStorage"
        ].includes(key)
      ) {
        continue;
      }

      let value;
      try {
        value = obj[key];
      } catch {
        continue;
      }

      if (value && (typeof value === "object" || typeof value === "function")) {
        varrer(value, `${path}.${key}`, depth + 1);
      }
    }
  }

  varrer(window.PDF_INSTANCES, "PDF_INSTANCES");
  varrer(window.PDFViewerApplication, "PDFViewerApplication");
  varrer(window.PDFController, "PDFController");
  varrer(window.pdf, "pdf");

  const unicos = [];
  const objs = new WeakSet();

  for (const c of candidatos) {
    if (!c.obj || objs.has(c.obj)) continue;
    objs.add(c.obj);
    unicos.push(c);
  }

  console.table(
    unicos.map((c, i) => ({
      i,
      path: c.path,
      numPages: c.obj?.numPages || null,
      hasGetData: typeof c.obj?.getData === "function",
      hasGetPage: typeof c.obj?.getPage === "function"
    }))
  );

  for (const candidato of unicos) {
    const ok = await tentarObjetoPdf(candidato.path, candidato.obj);
    if (ok) return;
  }

  log("4", "Nenhum objeto de memória funcionou. Tentando URLs carregadas pela página...");

  const urls = [...new Set(
    performance.getEntriesByType("resource")
      .map(r => r.name)
      .filter(u => /pdf|pdfViewer|document|download|file/i.test(u))
  )];

  console.table(urls.map((u, i) => ({ i, url: u })));

  for (const url of urls) {
    try {
      log("FETCH", `Tentando baixar: ${url}`);

      const response = await fetch(url, {
        credentials: "include",
        cache: "no-store",
        headers: {
          "Accept": "application/pdf,*/*",
          "Range": "bytes=0-"
        }
      });

      const buffer = await response.arrayBuffer();
      const bytes = new Uint8Array(buffer);

      console.table({
        url,
        status: response.status,
        contentType: response.headers.get("content-type"),
        contentLength: response.headers.get("content-length"),
        contentRange: response.headers.get("content-range"),
        bytes: bytes.byteLength,
        assinatura: assinatura(bytes)
      });

      if (parecePdf(bytes)) {
        log("SUCESSO", `PDF baixado via fetch: ${bytes.byteLength} bytes`);
        baixarBytes(bytes);
        return;
      }
    } catch (e) {
      warn("FETCH", `Falhou em ${url}`, e);
    }
  }

  erro(
    "FINAL",
    "Não consegui extrair automaticamente. Como último recurso, use Ctrl+P > Salvar como PDF."
  );
})();
