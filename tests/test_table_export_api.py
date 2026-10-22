import csv
import io

import pytest
from fastapi.testclient import TestClient
from openpyxl import load_workbook

from api.prediction_api import create_app


@pytest.mark.parametrize('fmt', ['xlsx', 'csv', 'tsv', 'txt', 'json'])
def test_browser_table_export_is_real_selected_format(fmt):
    result = TestClient(create_app()).post('/exports/table', json={
        'format': fmt, 'columns': ['tree_id', 'diameter'],
        'rows': [['=not_a_formula', None], ['树-002', 23.0]],
    })
    assert result.status_code == 200
    assert f'results.{fmt}' in result.headers['content-disposition']
    if fmt == 'xlsx':
        sheet = load_workbook(io.BytesIO(result.content)).active
        assert sheet['A2'].value == '=not_a_formula'
        assert sheet['A2'].data_type == 's'
        assert sheet['B2'].value is None
        assert sheet['B3'].value == 23.0
    elif fmt == 'json':
        assert result.json()[0]['diameter'] is None
        assert result.json()[1]['tree_id'] == '树-002'
    else:
        rows = list(csv.reader(io.StringIO(result.content.decode('utf-8-sig')),
                               delimiter=',' if fmt == 'csv' else '\t'))
        assert rows[1] == ['=not_a_formula', '']
        assert rows[2] == ['树-002', '23.0']


def test_invalid_browser_export_rejected():
    client = TestClient(create_app())
    for data in [dict(format='exe', columns=['id'], rows=[]),
                 dict(format='csv', columns=['id'], rows=[[1, 2]]),
                 dict(format='csv', columns=['id'], rows=[[{'a': 1}]])]:
        assert client.post('/exports/table', json=data).status_code == 422
