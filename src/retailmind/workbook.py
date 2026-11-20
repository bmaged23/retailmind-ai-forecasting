"""Streaming access to the two source workbook sheets."""
from openpyxl import load_workbook


def iter_rows(path):
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        if len(book.worksheets) != 2:
            raise ValueError('Expected exactly two workbook sheets')
        for sheet in book.worksheets:
            rows = sheet.iter_rows(values_only=True)
            headers = [str(value).strip() if value is not None else '' for value in next(rows)]
            required = {'StockCode', 'Quantity', 'InvoiceDate'}
            if not required.issubset(headers) or not ({'Invoice', 'InvoiceNo'} & set(headers)) or not ({'Price', 'UnitPrice'} & set(headers)):
                raise ValueError(f'Missing required columns in {sheet.title}')
            for values in rows:
                yield dict(zip(headers, values))
    finally:
        book.close()
