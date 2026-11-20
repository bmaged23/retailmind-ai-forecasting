from retailmind.family_audit import audit, similarity


def test_similarity_ignores_colour_but_does_not_assume_identity():
    assert similarity('BLUE GLASS BAG', 'RED GLASS BAG') == 1
    assert similarity('WOOD BOX', 'PAPER BAG') == 0
    assert similarity('RED', 'BLUE') is None


def test_audit_detects_case_collisions_without_merging():
    rows = [{'candidate_code_base': '1', 'stock_code': '1A', 'description': 'RED BOX'},
            {'candidate_code_base': '1', 'stock_code': '1a', 'description': 'BLUE BOX'},
            {'candidate_code_base': '2', 'stock_code': '2A', 'description': 'BAG'}]
    groups = audit(rows)
    assert len(groups) == 1 and groups[0]['n_products'] == 2
    assert groups[0]['case_only_collision'] is True
    assert groups[0]['codes'] == '1A | 1a'
