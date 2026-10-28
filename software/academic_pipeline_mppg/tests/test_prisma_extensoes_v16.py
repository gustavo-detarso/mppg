"""Regressões locais e sem rede para módulos incrementais PRISMA do MPPG.

Executar a partir da raiz do software:
    python -m unittest discover -s tests -p 'test_prisma_extensoes_v16.py' -v
"""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

SOFTWARE = Path(__file__).resolve().parents[1]
PIPELINE = SOFTWARE / 'app_bundle' / 'scripts' / 'pipeline'


def load(name: str, file: Path):
    if not file.is_file():
        raise AssertionError(f'Módulo PRISMA ausente: {file}')
    spec = importlib.util.spec_from_file_location(name, file)
    if spec is None or spec.loader is None:
        raise AssertionError(f'Não foi possível preparar importação: {file}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPrismaExtensoes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qualidade = load('prisma_qualidade_regressao', SOFTWARE / 'academic_pipeline' / 'prisma_qualidade_ia.py')
        cls.pubmed = load('prisma_pubmed_regressao', PIPELINE / 'prisma_pubmed_abstracts.py')
        cls.cobertura = load('prisma_cobertura_regressao', PIPELINE / 'prisma_cobertura.py')
        cls.triagem = load('prisma_triagem_regressao', PIPELINE / 'prisma_triagem_nativa.py')

    def test_01_cli_auditoria_ia_presente(self):
        entry = (SOFTWARE / 'academic_pipeline' / '__main__.py').read_text(encoding='utf-8')
        self.assertIn('--prisma-qualidade-ia', entry)
        self.assertIn('prisma_qualidade_ia', entry)

    def test_02_qualidade_preserva_fontes_e_consultas(self):
        data = [{'titulo': 'Governança de riscos', 'resumo': 'Resumo de teste',
                 'doi': 'https://doi.org/10.1234/abc', 'fontes': ['pubmed', 'openalex'],
                 'consultas_busca': ['pubmed:q1', 'openalex:q2']}]
        records = self.qualidade.records_normalized(data)
        self.assertEqual(records[0]['doi'], '10.1234/abc')
        self.assertEqual(set(records[0]['sources']), {'pubmed', 'openalex'})
        self.assertEqual(set(records[0]['strategies']), {'pubmed:q1', 'openalex:q2'})

    def test_03_qualidade_metrica_multifonte(self):
        records = self.qualidade.records_normalized([
            {'titulo': 'Teste', 'resumo': 'A', 'fontes': ['openalex', 'pubmed'], 'consultas_busca': ['q']},
            {'titulo': 'Outro', 'resumo': '', 'fontes': ['crossref'], 'consultas_busca': ['r']},
        ])
        stats = self.qualidade.metrics(records)
        self.assertEqual(stats['total_registros'], 2)
        self.assertEqual(stats['com_resumo'], 1)
        self.assertEqual(stats['com_multiplas_fontes'], 1)
        self.assertEqual(stats['sem_fonte'], 0)
        self.assertEqual(stats['sem_consulta'], 0)

    def test_04_pubmed_extrai_resumos_estruturados(self):
        xml = ('<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID>'
               '<Article><Abstract><AbstractText Label="BACKGROUND">A <i>B</i></AbstractText>'
               '<AbstractText Label="RESULTS">C</AbstractText></Abstract></Article>'
               '</MedlineCitation></PubmedArticle></PubmedArticleSet>')
        result = self.pubmed.extract_pubmed_abstracts(xml)
        self.assertEqual(result['123'], 'BACKGROUND: A B\nRESULTS: C')

    def test_05_pubmed_efetch_mockado_sem_rede(self):
        xml = ('<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID>'
               '<Article><Abstract><AbstractText>Texto</AbstractText></Abstract></Article>'
               '</MedlineCitation></PubmedArticle></PubmedArticleSet>')
        urls = []
        def responder(url, **kwargs):
            urls.append(url)
            return xml
        output, warnings = self.pubmed.fetch_pubmed_abstracts(['123', '123', 'invalido'], request_text=responder)
        self.assertEqual(output, {'123': 'Texto'})
        self.assertEqual(warnings, [])
        self.assertEqual(len(urls), 1)
        self.assertIn('efetch.fcgi', urls[0])

    def test_06_cobertura_desativada_por_padrao(self):
        _, s = self.cobertura._search_settings({'busca_prisma': {}})
        self.assertFalse(s['habilitada'])

    def test_07_cobertura_aplica_limites(self):
        _, s = self.cobertura._search_settings({'busca_prisma': {'cobertura': {
            'habilitada': True, 'limite_por_consulta': 999999,
            'tamanho_pagina': 999, 'max_paginas': 999}}})
        self.assertTrue(s['habilitada'])
        self.assertEqual(s['limite_por_consulta'], 10000)
        self.assertEqual(s['tamanho_pagina'], 100)
        self.assertEqual(s['max_paginas'], 200)

    def test_08_cobertura_estrategias_por_base(self):
        cfg = {'busca_prisma': {'estrategias': [{'consultas': ['q1'],
                'consultas_por_base': {'pubmed': ['q1[tiab]'], 'openalex': ['q1 adapted']}}]}}
        v = self.cobertura._strategy_variants(cfg)
        self.assertEqual(v['q1']['pubmed'], 'q1[tiab]')
        self.assertEqual(v['q1']['openalex'], 'q1 adapted')

    def test_09_cobertura_audita_sem_proveniencia(self):
        stats = self.cobertura._audit_records([
            {'titulo': 'A', 'resumo': 'Resumo', 'fontes': ['pubmed'], 'consultas_busca': ['q1']},
            {'titulo': 'B', 'resumo': '', 'fontes': [], 'consultas_busca': []},
        ])
        self.assertEqual(stats['registros_unicos'], 2)
        self.assertEqual(stats['com_resumo'], 1)
        self.assertEqual(stats['sem_fonte'], 1)
        self.assertEqual(stats['sem_consulta'], 1)

    def test_10_triagem_normaliza_proveniencia(self):
        self.assertEqual(self.triagem.provenance('pubmed;openalex'), ['openalex', 'pubmed'])
        self.assertEqual(self.triagem.provenance('["pubmed","openalex","pubmed"]'), ['openalex', 'pubmed'])

    def test_11_triagem_rejeita_ids_duplicados(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'triagem.csv'
            columns = ['id_registro', 'titulo', 'fontes', 'consultas_busca',
                       'decisao_titulo_resumo', 'decisao_texto_completo', 'incluir_final']
            with path.open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                row = dict(id_registro='registro-1', titulo='Titulo', fontes='pubmed',
                           consultas_busca='q1', decisao_titulo_resumo='PENDENTE',
                           decisao_texto_completo='NAO_INICIADO', incluir_final='PENDENTE')
                writer.writerow(row)
                writer.writerow(row)
            with self.assertRaises(ValueError):
                self.triagem.read_csv(path)

    def test_12_importador_nativo_preserva_contrato(self):
        import ast
        file = PIPELINE / 'prisma_busca_externa.py'
        tree = ast.parse(file.read_text(encoding='utf-8'))
        funcs = {x.name for x in tree.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.assertTrue({'_read_triage', '_decision', 'import_manual_prisma_triage'} <= funcs)


if __name__ == '__main__':
    unittest.main(verbosity=2)
