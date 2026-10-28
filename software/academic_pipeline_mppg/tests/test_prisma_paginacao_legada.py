"""Testes sem rede de paginacao PRISMA; fixtures sintéticas e falha fechada."""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit
import socket
import types

MODULE = Path(__file__).resolve().parent / 'prisma_paginacao_legada.py'
if not MODULE.is_file():
    MODULE = (Path(__file__).resolve().parents[1] / 'app_bundle'
              / 'scripts' / 'pipeline' / 'prisma_paginacao_legada.py')
spec = importlib.util.spec_from_file_location('prisma_paginacao_legada', MODULE)
pag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pag)


class MockProvider:
    def __init__(self, *, totals=None, stop=None, fail_page=None, without_total=False):
        self.totals = totals or {'scopus': 5, 'wos': 5, 'scielo': None}
        self.stop = stop
        self.fail_page = fail_page
        self.without_total = without_total
        self.requests = []
        self.ns = {'_mppg_cobertura_v16_instalada': True,
                   '_mppg_cobertura_v16_instalada_marker': True}
        self.ns['_request_json'] = self.request_json
        self.ns['_request_text'] = self.request_text
        self.ns['_scopus'] = self.native_scopus
        self.ns['_wos'] = self.native_wos
        self.ns['_scielo'] = self.native_scielo
        self.calls = []
        module = types.ModuleType('prisma_cobertura')
        module.CALLS = self.calls
        self.module = module

    def params(self, url):
        return parse_qs(urlsplit(url).query, keep_blank_values=True)

    def request_json(self, url, **kwargs):
        self.requests.append(url)
        params = self.params(url)
        base = 'scopus' if 'elsevier' in url else 'wos'
        page = int(params.get('start', ['0'])[0]) // max(1, int(params.get('count', ['1'])[0])) if base == 'scopus' else int(params.get('page', ['1'])[0])-1
        if self.fail_page == page:
            raise RuntimeError('FALHA FICTICIA')
        size = int(params.get('count', params.get('limit', ['10']))[0])
        total = self.totals[base]
        start = page * size
        count = max(0, min(size, total-start))
        if base == 'scopus':
            results = {'entry': [{'id': start + i, 'doi':f'10.5555/{start+i}', 'title':f'Title {start+i}'} for i in range(count)]}
            if not self.without_total: results['opensearch:totalResults'] = str(total)
            return {'search-results':results}
        results = {'hits':[{'id': start + i, 'doi':f'10.5555/{start+i}', 'title':f'Title {start+i}'} for i in range(count)]}
        if not self.without_total: results['metadata']={'total':total,'limit':size,'page':page+1}
        return results

    def request_text(self, url, **kwargs):
        self.requests.append(url)
        params = self.params(url)
        page = int(params.get('page', ['1'])[0])-1
        if self.fail_page == page: raise RuntimeError('FALHA FICTICIA')
        return 'SCIELO_' + str(page)

    def native_scopus(self, query, config):
        url = 'https://api.elsevier.com/content/search/scopus?query='+query+'&count='+str(min(config['per_base'],config['scopus_per_query']))
        payload = self.ns['_request_json'](url)
        items = payload['search-results']['entry']
        return ([{'id_registro':'scopus:'+str(x['id']),'doi':x['doi'],'titulo':x['title']} for x in items],
                {'provider':'scopus','retrieved':len(items),'url':url})

    def native_wos(self, query, config):
        url = 'https://api.clarivate.com/apis/wos-starter/v1/documents?q='+query+'&limit='+str(config['per_base'])+'&page=1'
        payload = self.ns['_request_json'](url)
        items = payload['hits']
        return ([{'id_registro':'wos:'+str(x['id']),'doi':x['doi'],'titulo':x['title']} for x in items],
                {'provider':'wos','retrieved':len(items),'url':url})

    def native_scielo(self, query, config):
        url = ('https://search.scielo.org/?q='+query+'&format=summary&count='+str(config['per_base'])
               +'&page=1&from=0&filter%5Bin%5D%5B%5D=scl&filter%5Btype%5D%5B%5D=article')
        payload = self.ns['_request_text'](url)
        page = int(payload.rsplit('_',1)[1])
        start = page * config['per_base']
        total = self.stop if self.stop is not None else 5
        items = [{'id_registro':'scielo:'+str(i+1),'titulo':f'Titulo {start+i}', 'url':f'https://scielo.org/{start+i}'}
                 for i in range(max(0,min(config['per_base'],total-start)))]
        return items,{'provider':'scielo','retrieved':len(items),'url':url}

    def installed(self):
        with patch.dict(sys.modules, {'prisma_cobertura':self.module}):
            pag.install_hooks(self.ns)
        return self

    def config(self, *, cap=20, size=2, pages=10, coverage=True):
        return {'per_base': 10,'scopus_per_query': 10,
                '__mppg_cobertura__': {'habilitada': coverage,
                    'limite_por_consulta':cap,'tamanho_pagina':size,'max_paginas':pages},
                '__mppg_variants__':{'base':{'scopus':'SCOPUS_QUERY','wos':'WOS_QUERY','scielo':'SCIELO_QUERY'}}}


class TestPaginacao(TestCase):
    def test_01_rewrite_scopus(self):
        url=pag._rewrite('https://api.elsevier.com/content/search/scopus?query=abc&count=10','scopus',2,25)
        self.assertEqual(parse_qs(urlsplit(url).query),{'query':['abc'],'start':['50'],'count':['25']})

    def test_02_rewrite_wos(self):
        url=pag._rewrite('https://api.clarivate.com/apis/wos-starter/v1/documents?q=abc&limit=80&page=1','wos',2,50)
        self.assertEqual(parse_qs(urlsplit(url).query),{'q':['abc'],'limit':['50'],'page':['3'],'db':['WOS']})

    def test_03_rewrite_scielo_sem_filtros(self):
        url=pag._rewrite('https://search.scielo.org/?q=x&format=summary&filter%5Bin%5D%5B%5D=scl&filter%5Btype%5D%5B%5D=article','scielo',2,15)
        p=parse_qs(urlsplit(url).query)
        self.assertNotIn('filter[in][]',p)
        self.assertNotIn('filter[type][]',p)
        self.assertEqual(p['from'],['30']);self.assertEqual(p['page'],['3'])

    def test_04_scopus_tres_paginas_total(self):
        m=MockProvider().installed()
        with patch.object(pag.time,'sleep',return_value=None),patch.object(socket,'socket',side_effect=AssertionError('rede')):
            rows,log=m.ns['_scopus']('base',m.config())
        self.assertEqual(len(rows),5)
        self.assertEqual(log['pages_fetched'],3)
        self.assertTrue(log['coverage_exhausted'])
        self.assertEqual(log['query_effective'],'SCOPUS_QUERY')
        self.assertEqual([int(m.params(x)['start'][0]) for x in m.requests],[0,2,4])
        self.assertEqual(len(m.calls),1)

    def test_05_wos_tres_paginas_db_limite(self):
        m=MockProvider().installed()
        with patch.object(pag.time,'sleep',return_value=None):
            rows,log=m.ns['_wos']('base',m.config())
        self.assertEqual(len(rows),5)
        self.assertTrue(log['coverage_exhausted'])
        self.assertEqual([int(m.params(x)['page'][0]) for x in m.requests],[1,2,3])
        self.assertEqual({m.params(x)['db'][0] for x in m.requests},{'WOS'})

    def test_06_scielo_tres_paginas_total_desconhecido(self):
        m=MockProvider().installed()
        with patch.object(pag.time,'sleep',return_value=None):
            rows,log=m.ns['_scielo']('base',m.config())
        self.assertEqual(len(rows),5)
        self.assertTrue(log['coverage_truncated'])
        self.assertFalse(log['coverage_exhausted'])
        self.assertEqual(len({r['id_registro'] for r in rows}),5)
        self.assertEqual([int(m.params(x)['from'][0]) for x in m.requests],[0,2,4])
        self.assertTrue(all('filter[in][]' not in m.params(x) for x in m.requests))

    def test_07_disabled_no_change(self):
        m=MockProvider().installed()
        rows,log=m.ns['_scielo']('base',m.config(coverage=False))
        self.assertEqual(len(rows),5)
        self.assertIn('filter[in][]',m.params(m.requests[0]))
        self.assertNotIn('coverage_exhausted',log)

    def test_08_limited_results_reported_truncated(self):
        m=MockProvider().installed()
        rows,log=m.ns['_wos']('base',m.config(cap=3,size=2))
        self.assertEqual(len(rows),3)
        self.assertTrue(log['coverage_truncated'])
        self.assertEqual(log['source_total_available'],5)

    def test_09_first_page_failure_propagates(self):
        m=MockProvider(fail_page=0).installed()
        with self.assertRaisesRegex(RuntimeError,'FALHA FICTICIA'):
            m.ns['_scopus']('base',m.config())
        self.assertEqual(m.calls,[])
        self.assertIs(m.ns['_request_json'].__func__,m.request_json.__func__)

    def test_10_second_page_failure_partial_report(self):
        m=MockProvider(fail_page=1).installed()
        with patch.object(pag.time,'sleep',return_value=None):
            rows,log=m.ns['_wos']('base',m.config())
        self.assertEqual(len(rows),2)
        self.assertTrue(log['coverage_truncated'])
        self.assertIn('Falha na página 2',' '.join(log['coverage_warnings']))

    def test_11_no_total_never_complete(self):
        m=MockProvider(without_total=True).installed()
        with patch.object(pag.time,'sleep',return_value=None):
            rows,log=m.ns['_scopus']('base',m.config())
        self.assertEqual(len(rows),5)
        self.assertTrue(log['coverage_truncated'])
        self.assertIsNone(log['source_total_available'])

    def test_12_max_pages_caps(self):
        m=MockProvider().installed()
        rows,log=m.ns['_wos']('base',m.config(pages=1))
        self.assertEqual(len(rows),2)
        self.assertTrue(log['coverage_truncated'])

    def test_13_no_url_rewrite_unrelated(self):
        self.assertFalse(pag._matches('https://example.com/?q=abc','scielo'))
        self.assertFalse(pag._matches('http://search.scielo.org/?q=x&format=summary&filter%5Bin%5D%5B%5D=scl&filter%5Btype%5D%5B%5D=article','scielo'))
        self.assertFalse(pag._matches('https://evil.example.com/content/search/scopus','scopus'))

    def test_14_zero_results_complete_when_total_zero(self):
        m=MockProvider(totals={'scopus':0,'wos':0,'scielo':None}).installed()
        rows,log=m.ns['_scopus']('base',m.config())
        self.assertEqual(rows,[])
        self.assertTrue(log['coverage_exhausted'])
        self.assertEqual(log['source_total_available'],0)

    def test_15_install_twice_idempotent(self):
        m=MockProvider().installed()
        f=m.ns['_scopus']
        with patch.dict(sys.modules, {'prisma_cobertura':m.module}):
            pag.install_hooks(m.ns)
        self.assertIs(m.ns['_scopus'],f)

    def test_16_requires_coverage_installation(self):
        m=MockProvider()
        m.ns.pop('_mppg_cobertura_v16_instalada')
        with self.assertRaisesRegex(RuntimeError,'cobertura'):
            pag.install_hooks(m.ns)

    def test_17_scopus_max_page_size_25(self):
        m=MockProvider(totals={'scopus':21,'wos':0,'scielo':None}).installed()
        rows,log=m.ns['_scopus']('base',m.config(size=100,cap=25))
        self.assertLessEqual(int(m.params(m.requests[0])['count'][0]),25)

    def test_18_wos_max_page_size_50(self):
        m=MockProvider(totals={'scopus':0,'wos':21,'scielo':None}).installed()
        rows,log=m.ns['_wos']('base',m.config(size=100,cap=50))
        self.assertLessEqual(int(m.params(m.requests[0])['limit'][0]),50)

    def test_19_scielo_identidade_estavel_por_url(self):
        m=MockProvider(stop=4).installed()
        with patch.object(pag.time,'sleep',return_value=None):
            rows,log=m.ns['_scielo']('base',m.config(size=2))
        self.assertEqual(len({r['id_registro'] for r in rows}),4)
        self.assertTrue(all(r['id_registro'].startswith('scielo:') for r in rows))

    def test_20_exception_does_not_leak_raw_error_log(self):
        m=MockProvider(fail_page=1).installed()
        with patch.object(pag.time,'sleep',return_value=None):
            rows,log=m.ns['_scopus']('base',m.config(size=2))
        self.assertNotIn('FALHA FICTICIA',' '.join(log['coverage_warnings']))

    def test_21_sanitize_logged_url_institutional_credentials(self):
        secret = 'SENSITIVO_NAO_LOGAR'
        url = ('https://nome:senha@api.clarivate.com/apis/wos-starter/v1/documents'
               '?q=TS%3D%28risk%29&api_key=' + secret
               + '&access_token=' + secret + '&email=private%40example.com')
        sanitized = pag._safe_log_url(url)
        self.assertNotIn(secret, sanitized)
        self.assertNotIn('senha', sanitized)
        self.assertNotIn('private%40example.com', sanitized)
        self.assertIn('q=TS%3D%28risk%29', sanitized)
        self.assertEqual(parse_qs(urlsplit(sanitized).query)['api_key'], ['REDACTED'])

    def test_22_actual_coverage_log_never_leaks_token(self):
        m = MockProvider().installed()
        secret = 'CHAVE_NAO_MOSTRAR'
        def original(query, config):
            url = ('https://api.clarivate.com/apis/wos-starter/v1/documents'
                   '?q=abc&limit=2&page=1&token=' + secret)
            payload = m.ns['_request_json'](url)
            records = [{'id_registro': 'wos:'+str(x['id']), 'doi': x['doi']}
                       for x in payload['hits']]
            return records, {'provider':'wos','url':url,'retrieved':len(records)}
        m.ns['_wos'] = original
        # Reinstala wrapper na instancia que possui funcao ficticia.
        m.ns.pop('_mppg_legacy_paging_v2_instalada')
        with patch.dict(sys.modules, {'prisma_cobertura':m.module}):
            pag.install_hooks(m.ns)
        with patch.object(pag.time, 'sleep', return_value=None):
            rows, log = m.ns['_wos']('base', m.config(cap=2, size=2))
        self.assertEqual(len(rows), 2)
        self.assertNotIn(secret, log['url'])
        self.assertEqual(parse_qs(urlsplit(log['url']).query)['token'], ['REDACTED'])

    def test_23_wos_rate_limit_across_separate_queries(self):
        m = MockProvider().installed()
        with patch.object(pag.time, 'monotonic', return_value=100.0), \
             patch.object(pag.time, 'sleep', return_value=None) as sleeper:
            previous = pag._WOS_LAST_REQUEST_AT
            try:
                pag._WOS_LAST_REQUEST_AT = None
                m.ns['_wos']('base', m.config(cap=1, size=1, pages=1))
                m.ns['_wos']('base', m.config(cap=1, size=1, pages=1))
                self.assertTrue(any(call.args and call.args[0] >= 1.0
                                    for call in sleeper.call_args_list))
            finally:
                pag._WOS_LAST_REQUEST_AT = previous

    def test_24_second_page_antibot_stops_provider(self):
        class BlockedByProvider(Exception):
            pass
        m = MockProvider().installed()
        m.ns['ProviderUnavailableError'] = BlockedByProvider
        old = m.ns['_request_text']
        def blocking_text(url, **kwargs):
            if int(parse_qs(urlsplit(url).query).get('page', ['1'])[0]) == 2:
                raise BlockedByProvider('Challenge detected')
            return old(url, **kwargs)
        m.ns['_request_text'] = blocking_text
        with patch.object(pag.time, 'sleep', return_value=None):
            with self.assertRaises(BlockedByProvider):
                m.ns['_scielo']('base', m.config())

if __name__ == '__main__':
    main(verbosity=2)
