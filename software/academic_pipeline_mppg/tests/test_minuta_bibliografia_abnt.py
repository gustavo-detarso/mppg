import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from academic_pipeline import minuta_runtime as mod


class TestMinutaBibliografiaABNT(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        self.base = Path(self.tmp.name)
        self.pdf = self.base / "prova.pdf"
        self.org = self.base / "prova.org"
        self.bbl = self.base / "prova.bbl"

        self.pdf.write_bytes(b"%PDF-1.4\n")

        self.org.write_text(
            "* Tema\n"
            "Estudo demonstrativo [cite:@silva2026].\n",
            encoding="utf-8",
        )

        self.bbl.write_text(
            "\\entry{silva2026}{article}{}\n",
            encoding="utf-8",
        )

    def validar(self, texto_pdf):
        resultado = subprocess.CompletedProcess(
            args=["pdftotext"],
            returncode=0,
            stdout=texto_pdf,
            stderr="",
        )

        with patch.object(
            mod.subprocess, "run", return_value=resultado
        ):
            mod._mppg_finalize_bib_pdf(
                self.pdf, self.base
            )

    def test_01_referencia_arbitraria_valida(self):
        self.validar(
            "Texto demonstrativo (Silva, 2026).\n"
            "REFERÊNCIAS\n"
            "SILVA, Ana. Uma obra qualquer.\n"
        )

    def test_02_referencia_ausente_no_bbl(self):
        self.bbl.unlink()

        with self.assertRaises(RuntimeError):
            mod._mppg_finalize_bib_pdf(
                self.pdf, self.base
            )

    def test_03_chave_nao_resolvida(self):
        self.org.write_text(
            "* Tema\n"
            "Texto [cite:@outra2026].\n",
            encoding="utf-8",
        )

        with self.assertRaises(RuntimeError):
            mod._mppg_finalize_bib_pdf(
                self.pdf, self.base
            )

    def test_04_secao_bibliografica_ausente(self):
        with self.assertRaises(RuntimeError):
            self.validar(
                "Texto sobre referencias em pesquisas.\n"
            )

    def test_05_multiplas_citacoes(self):
        self.org.write_text(
            "* Tema\n"
            "Texto [cite:@silva2026;@souza2025].\n",
            encoding="utf-8",
        )

        self.bbl.write_text(
            "\\entry{silva2026}{article}{}\n"
            "\\entry{souza2025}{book}{}\n",
            encoding="utf-8",
        )

        self.validar(
            "Texto demonstrativo.\n"
            "REFERÊNCIAS\n"
            "SILVA, Ana. Obra A.\n"
            "SOUZA, João. Obra B.\n"
        )


    def test_06_recompilacao_recupera_bibliografia(self):
        self.bbl.unlink()

        self.pdf.with_suffix(".tex").write_text(
            "% LaTeX de teste\n",
            encoding="utf-8",
        )

        commands = []

        def fake_run(args, **kwargs):
            commands.append(args[0])

            if args[0] == "latexmk":
                self.bbl.write_text(
                    r"\entry{silva2026}{article}{}" + "\n",
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(
                    args, 0, stdout="latexmk simulado\n"
                )

            if args[0] == "pdftotext":
                return subprocess.CompletedProcess(
                    args,
                    0,
                    stdout=(
                        "Texto demonstrativo (Silva, 2026).\n"
                        "REFERÊNCIAS\n"
                        "SILVA, Ana. Obra demonstrativa.\n"
                    ),
                    stderr="",
                )

            raise AssertionError(
                f"Comando inesperado: {args[0]}"
            )

        with (
            patch.object(
                mod.subprocess, "run", side_effect=fake_run
            ),
            patch.object(
                mod.shutil, "which", return_value="latexmk"
            ),
        ):
            mod._mppg_finalize_bib_pdf(
                self.pdf, self.base
            )

        self.assertEqual(
            commands, ["latexmk", "pdftotext"]
        )

        self.assertTrue(
            (
                self.base / "compilacao_biber_mppg.log"
            ).is_file()
        )

    def test_07_recompilacao_falha_controlada(self):
        self.bbl.unlink()

        self.pdf.with_suffix(".tex").write_text(
            "% LaTeX de teste\n",
            encoding="utf-8",
        )

        resultado = subprocess.CompletedProcess(
            args=["latexmk"],
            returncode=2,
            stdout="Erro simulado de compilacao\n",
        )

        with (
            patch.object(
                mod.subprocess,
                "run",
                return_value=resultado,
            ) as run_mock,
            patch.object(
                mod.shutil,
                "which",
                return_value="latexmk",
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "BibLaTeX/Biber falhou"
            ):
                mod._mppg_finalize_bib_pdf(
                    self.pdf, self.base
                )

        run_mock.assert_called_once()

        log = self.base / "compilacao_biber_mppg.log"

        self.assertTrue(log.is_file())
        self.assertIn(
            "Erro simulado",
            log.read_text(encoding="utf-8"),
        )


    def test_08_rejeita_parenteses_nao_resolvidos(self):
        with self.assertRaisesRegex(
            RuntimeError, "PDF sem bibliografia"
        ):
            self.validar(
                "Texto com citacao (??).\n"
                "REFERÊNCIAS\n"
                "SILVA, Ana. Obra demonstrativa.\n"
            )

    def test_09_rejeita_colchetes_nao_resolvidos(self):
        with self.assertRaisesRegex(
            RuntimeError, "PDF sem bibliografia"
        ):
            self.validar(
                "Texto com citacao [??].\n"
                "REFERÊNCIAS\n"
                "SILVA, Ana. Obra demonstrativa.\n"
            )

    def test_10_rejeita_org_cite_literal(self):
        with self.assertRaisesRegex(
            RuntimeError, "PDF sem bibliografia"
        ):
            self.validar(
                "Texto [cite:@silva2026].\n"
                "REFERÊNCIAS\n"
                "SILVA, Ana. Obra demonstrativa.\n"
            )

    def test_11_preserva_perguntas_normais(self):
        self.validar(
            "Qual é o problema de pesquisa?\n"
            "Segundo Silva (2026), existe uma resposta.\n"
            "REFERÊNCIAS\n"
            "SILVA, Ana. Obra demonstrativa.\n"
        )


if __name__ == "__main__":
    unittest.main()
