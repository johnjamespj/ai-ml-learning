import numpy as np
import pytest

def test_kmeans_final_labels_match_returned_centers():
    from labs.lab03_kmeans_pca.solution import kmeans,assign_clusters
    X=np.random.default_rng(44).normal(size=(100,3))
    labels,centers=kmeans(X,4,steps=1,seed=8)
    np.testing.assert_array_equal(labels,assign_clusters(X,centers))

def test_attention_rejects_fully_masked_rows():
    torch=pytest.importorskip('torch')
    from labs.lab07_attention.solution import scaled_dot_product_attention
    q=torch.ones(1,1,2,3)
    with pytest.raises(ValueError): scaled_dot_product_attention(q,q,q,torch.zeros(2,2,dtype=torch.bool))

def test_student_workspace_never_copies_references(tmp_path):
    from pathlib import Path
    import subprocess,sys
    root=Path(__file__).resolve().parents[1]
    destination=tmp_path/'student'
    subprocess.run([sys.executable,str(root/'tools/make_student_workspace.py'),'--output',str(destination)],check=True)
    assert len(list((destination/'labs').glob('*/solution.py')))==12
    assert not list(destination.rglob('reference_solutions.py'))
    assert 'raise NotImplementedError' in (destination/'labs/lab01_linear_regression/solution.py').read_text()
