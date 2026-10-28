"""Regressoes do adaptador Scopus, integralmente sem rede."""

from __future__ import annotations

import socket
import unittest

from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from app_bundle.scripts.pipeline import prisma_busca_externa as prisma


class TestScopusQueryFGV(unittest.TestCase):

    def _call(self, query, *, env=None, config=None):
        captured = []

        secrets = (
            {"SCOPUS_API_KEY": "CHAVE_FICTICIA"}
            if env is None else env
        )

        def fake_request(url, **kwargs):
            captured.append((url, kwargs.get("headers", {})))
            return {"search-results": {"entry": []}}

        config = config or {
            "per_base": 20,
            "scopus_per_query": 10,
        }

        with (
            patch.object(
                prisma, "_env",
                side_effect=lambda name: secrets.get(name, ""),
            ),
            patch.object(
                prisma, "_request_json",
                side_effect=fake_request,
            ),
            patch.object(
                socket, "socket",
                side_effect=AssertionError("Rede proibida"),
            ),
        ):
            records, log = prisma._scopus(query, config)

        self.assertEqual(records, [])
        self.assertEqual(log["provider"], "scopus")
        self.assertEqual(len(captured), 1)

        url, headers = captured[0]

        return parse_qs(urlsplit(url).query), headers

    def test_01_preserva_and(self):
        query = '"risk governance" AND "disability assessment"'
        params, _ = self._call(query)

        self.assertEqual(
            params["query"][0],
            "TITLE-ABS-KEY(" + query + ")",
        )

    def test_02_preserva_or(self):
        query = '"risk governance" OR "risk management"'
        params, _ = self._call(query)

        self.assertEqual(
            params["query"][0],
            "TITLE-ABS-KEY(" + query + ")",
        )

    def test_03_preserva_grupos_and_or_not(self):
        query = (
            '("risk governance" OR "public accountability") '
            'AND "disability assessment" NOT "animal study"'
        )

        params, _ = self._call(query)

        self.assertEqual(
            params["query"][0],
            "TITLE-ABS-KEY(" + query + ")",
        )

    def test_04_nao_duplica_campo_existente(self):
        query = (
            'TITLE-ABS-KEY('
            '"risk governance" AND "disability assessment"'
            ')'
        )

        params, _ = self._call(query)

        self.assertEqual(params["query"][0], query)

    def test_05_preserva_campos_especificos(self):
        query = (
            'TITLE("disability benefits") '
            'AND ABS("administrative review")'
        )

        params, _ = self._call(query)

        self.assertEqual(params["query"][0], query)

    def test_06_consulta_simples(self):
        params, _ = self._call("disability")

        self.assertEqual(
            params["query"][0],
            "TITLE-ABS-KEY(disability)",
        )

    def test_07_rejeita_consulta_vazia(self):
        with self.assertRaisesRegex(ValueError, "vazia"):
            self._call("   ")

    def test_08_rejeita_chave_ausente(self):
        with self.assertRaisesRegex(RuntimeError, "SCOPUS_API_KEY"):
            self._call(
                '"disability benefits"',
                env={},
            )

    def test_09_preserva_credenciais_e_limite(self):
        params, headers = self._call(
            '"disability benefits" AND audit',
            env={
                "SCOPUS_API_KEY": "CHAVE_FICTICIA",
                "SCOPUS_INSTTOKEN": "TOKEN_FICTICIO",
            },
            config={
                "per_base": 20,
                "scopus_per_query": 7,
            },
        )

        self.assertEqual(params["count"], ["7"])

        self.assertEqual(
            headers["X-ELS-APIKey"],
            "CHAVE_FICTICIA",
        )

        self.assertEqual(
            headers["X-ELS-Insttoken"],
            "TOKEN_FICTICIO",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
