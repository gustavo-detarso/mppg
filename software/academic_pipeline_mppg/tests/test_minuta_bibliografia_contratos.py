import tempfile
import unittest
from pathlib import Path

from academic_pipeline import minuta_runtime as mod


class TestBibliografiaContratos(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        self.base = Path(self.tmp.name)
        self.bib = self.base / "referencias.bib"
        self.csl = self.base / "abnt-autor-data-mppg.csl"

        self.bib.write_text(
            "@book{silva2026,\n"
            "author={Silva, Ana},\n"
            "title={Livro de teste},\n"
            "year={2026}}\n",
            encoding="utf-8",
        )

        self.csl.write_text(
            "<style/>", encoding="utf-8"
        )

    def config(self):
        return {
            "bibliografia": {
                "habilitada": True,
                "arquivo": self.bib.name,
                "csl": self.csl.name,
                "estilo": "abnt",
                "backend": "biber",
            }
        }

    def test_01_configuracao_valida(self):
        result = mod._mppg_bib_config(
            self.config(), self.base
        )
        self.assertEqual(result, (self.bib, self.csl))

    def test_02_bibliografia_desativada(self):
        cfg = {"bibliografia": {"habilitada": False}}
        self.assertIsNone(
            mod._mppg_bib_config(cfg, self.base)
        )

    def test_03_tabela_invalida(self):
        with self.assertRaises(ValueError):
            mod._mppg_bib_config(
                {"bibliografia": []}, self.base
            )

    def test_04_habilitada_nao_booleana(self):
        cfg = self.config()
        cfg["bibliografia"]["habilitada"] = "true"

        with self.assertRaises(ValueError):
            mod._mppg_bib_config(cfg, self.base)

    def test_05_backend_invalido(self):
        cfg = self.config()
        cfg["bibliografia"]["backend"] = "bibtex"

        with self.assertRaises(ValueError):
            mod._mppg_bib_config(cfg, self.base)

    def test_06_bib_ausente(self):
        self.bib.unlink()

        with self.assertRaises(FileNotFoundError):
            mod._mppg_bib_config(
                self.config(), self.base
            )

    def test_07_csl_ausente(self):
        self.csl.unlink()

        with self.assertRaises(FileNotFoundError):
            mod._mppg_bib_config(
                self.config(), self.base
            )

    def test_08_extensao_invalida(self):
        destino = self.base / "referencias.txt"
        self.bib.rename(destino)

        cfg = self.config()
        cfg["bibliografia"]["arquivo"] = destino.name

        with self.assertRaises(ValueError):
            mod._mppg_bib_config(cfg, self.base)

    def test_09_link_simbolico_interno_rejeitado(self):
        destino = self.base / "referencias_originais.bib"
        self.bib.rename(destino)
        self.bib.symlink_to(destino.name)

        with self.assertRaises(
            (ValueError, FileNotFoundError)
        ):
            mod._mppg_bib_config(
                self.config(), self.base
            )

    def test_10_minuta_antiga_sem_bibliografia(self):
        source = self.base / "conteudo.org"

        source.write_text(
            "* Tema\n"
            "Texto sem referencias bibliograficas.\n\n"
            "* Problema\n"
            "Problema demonstrativo.\n\n"
            "* Protocolo da Revisao\n"
            "Procedimento de teste.\n",
            encoding="utf-8",
        )

        cfg = {
            "_internal": {"nome": "teste_legado"},
            "documento": {
                "titulo_trabalho": "Documento legado",
                "autor": "Autor de Teste",
            },
        }

        rendered, _ = mod._render(
            cfg, source, self.base / "saida_legada"
        )

        content = rendered.read_text(
            encoding="utf-8"
        )

        self.assertNotIn(
            "#+CITE_EXPORT:", content
        )
        self.assertNotIn(
            "#+PRINT_BIBLIOGRAPHY:", content
        )


    def test_11_diretorio_simbolico_rejeitado(self):
        real = self.base / "bibliografia_real"
        real.mkdir()

        self.bib.rename(real / self.bib.name)

        link = self.base / "bibliografia_link"
        link.symlink_to(real.name, target_is_directory=True)

        cfg = self.config()
        cfg["bibliografia"]["arquivo"] = (
            "bibliografia_link/referencias.bib"
        )

        with self.assertRaises(ValueError):
            mod._mppg_bib_config(cfg, self.base)

    def test_12_link_simbolico_csl_rejeitado(self):
        original = self.base / "estilo_original.csl"
        self.csl.rename(original)
        self.csl.symlink_to(original.name)

        with self.assertRaises(ValueError):
            mod._mppg_bib_config(
                self.config(), self.base
            )


if __name__ == "__main__":
    unittest.main()
