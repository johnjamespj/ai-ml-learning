import pytest
from assessment.checks import check_submission
from assessment.specs import EXAMS
from instructor import reference_solutions as ref

@pytest.mark.parametrize('exam',list(EXAMS))
def test_reference_meets_numerical_contract(exam):
    result=check_submission(exam,vars(ref))
    assert result['coding_checks']=='passed' and result['mastery']=='not_yet_assessed'

@pytest.mark.parametrize('exam',list(EXAMS))
def test_missing_submission_is_not_a_pass(exam):
    with pytest.raises(AssertionError): check_submission(exam,{})
