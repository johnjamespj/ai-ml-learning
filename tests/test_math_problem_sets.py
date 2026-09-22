import pytest
from assessment.math_checks import check
from instructor import math_reference as m
CASES=[('01_matrix_derivatives',m.gradient),('02_likelihood_map',m.map_coin),('03_entropy_kl',m.kl),
       ('04_pca_proof',m.principal_direction),('05_backprop',m.affine_backward),('06_attention_scaling',m.stable_softmax),
       ('07_bellman',m.bellman_backup),('08_detection',m.colored_matched_score)]
@pytest.mark.parametrize('name,fn',CASES)
def test_reference_math(name,fn): assert check(name,fn)['numerical_check']=='passed'
