"""Build mathematical problem sets with proofs, numerical checks and counterexamples."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
PROBLEMS=[
('01_matrix_derivatives','Matrix derivatives','02_calculus_matrix_calculus',
 'Derive the MSE gradient with differentials and trace identities. Derive the Hessian and its positive-semidefinite property. State when it is positive definite. Repeat with an explicit unpenalized intercept.',
 'gradient(X,y,w)','Vary finite-difference epsilon over 1e-2 through 1e-10. Plot relative gradient error and explain truncation versus cancellation.'),
('02_likelihood_map','Likelihood, MLE and MAP','04_statistics_likelihood',
 'Derive Bernoulli MLE and the Beta-Bernoulli posterior. Derive the interior posterior mode and specify when a boundary mode replaces that formula. Explain why posterior mean, MAP and MLE need not coincide.',
 'map_coin(heads,tails,alpha,beta)','Compare estimates and posterior uncertainty after 0, 5, 20 and 200 observations. Test an informative prior and a nearly boundary posterior.'),
('03_entropy_kl','Entropy, cross-entropy and KL','05_information_theory',
 'Prove H(p,q)=H(p)+KL(p||q) and nonnegativity with Jensen or the log inequality. Give a numerical counterexample to symmetry. Handle p_i=0 and q_i=0 explicitly.',
 'kl(p,q)','Compare confident-correct and confident-wrong classifiers with log loss. Never clip away the mathematical distinction between an infinite loss and a large finite loss.'),
('04_pca_proof','PCA, eigenvalues and SVD','09_pca_svd_kernels',
 'Use a Lagrange multiplier to derive the leading PCA direction. Explain sign ambiguity, covariance normalization and the rank-r reconstruction objective. Construct a classification task where the label lies in a low-variance direction.',
 'principal_direction(centered_X)','Sweep retained rank and plot reconstruction error. Compare PCA with and without feature standardization and state which metric each optimizes.'),
('05_backprop','Backpropagation and Jacobians','10_neural_network_math',
 'Derive the affine-layer weight and bias gradients under row-batched convention. Derive the chain rule for two layers and ReLU, then the residual Jacobian I+J_F. Distinguish a ReLU kink from an implementation bug.',
 'affine_backward(activations,upstream)','Check gradients away from kinks, then deliberately cross a kink. Compare a double batch-size division bug with a correct mean-loss gradient.'),
('06_attention_scaling','Attention variance and stable probabilities','11_attention_transformers',
 'Derive Var(q dot k)=d only under the stated independence and unit-variance assumptions. Construct correlated components that violate it. Prove row-wise softmax translation invariance. Explain fully masked rows.',
 'stable_softmax(scores)','Plot entropy across dimensions with and without sqrt(d) scaling; repeat with correlated queries/keys and explain why scaling alone may not fix saturation.'),
('07_bellman','Bellman equations and contraction','13_reinforcement_learning_math',
 'Derive the Bellman optimality operator and prove its sup-norm contraction for gamma<1 using row-stochastic transitions. Explain where the argument breaks with unrestricted function approximation. Distinguish termination from time-limit truncation.',
 'bellman_backup(V,P,R,gamma)','Compare exact value iteration with tabular Q-learning over multiple seeds. Do not use a time-limit flag to zero the bootstrap unless time remaining belongs to the modeled state.'),
('08_detection','Likelihood-ratio detection and whitening','14_signal_detection_math',
 'Derive the white- and colored-Gaussian known-signal likelihood ratios. Show the whitening equivalence with C=LL^T. Derive the unknown-phase quadrature statistic under orthogonal normalized templates, and state its limitations.',
 'colored_matched_score(x,s,C)','Estimate fresh-test Pd/Pfa with Wilson intervals. Sweep SNR while holding a noise-calibrated threshold fixed. Include out-of-distribution colored noise and explain why zero observed false alarms is not zero probability.')]

def main():
    for key,title,companion,proof,signature,experiment in PROBLEMS:
        starter=f'def {signature}:\n    raise NotImplementedError("Derive first, then implement")'
        function=signature.split('(')[0]
        cells=[nbf.v4.new_markdown_cell(f'# Problem set: {title}\n\n[Math companion](../{companion}.ipynb)\n\n## Proof and assumptions\n\n{proof}'),
               nbf.v4.new_markdown_cell('**Your derivation:**'),
               nbf.v4.new_code_cell("from pathlib import Path\nimport os,sys\nROOT=Path(os.environ.get('COURSE_ROOT',next((str(p) for p in [Path.cwd(),*Path.cwd().parents] if (p/'assessment').is_dir()),'.')))\nsys.path.insert(0,str(ROOT))\nimport numpy as np\nRUN_CHECKS=False"),
               nbf.v4.new_code_cell(starter),
               nbf.v4.new_code_cell(f"if RUN_CHECKS:\n    from assessment.math_checks import check\n    print(check('{key}', {function}))\nelse:\n    print('NOT GRADED: implement your result and enable RUN_CHECKS.')"),
               nbf.v4.new_markdown_cell('## Experiment and counterexample\n\n'+experiment+'\n\nRecord seeds, assumptions, measured errors, and what the check does not prove.'),
               nbf.v4.new_code_cell('# Add your experiment here. Do not replace a derivation with an unexplained library call.'),
               nbf.v4.new_markdown_cell('## Completion rule\n\nPass the numerical check, provide a correct derivation with assumptions, construct one counterexample, and explain the observed error. Numerical checks alone do not establish completion.')]
        nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'assessment':{'id':key,'status':'not_graded'}})
        target=ROOT/'math/problem_sets'/f'{key}.ipynb';target.parent.mkdir(parents=True,exist_ok=True);nbf.write(nb,target)
    print(f'Built {len(PROBLEMS)} mathematical problem sets')
if __name__=='__main__':main()
