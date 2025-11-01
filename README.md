<h1 align='center'>Operator Learning for Schrödinger Equation: Unitarity, Error Bounds, and Time Generalization</h1>

<div align='center'>
    <a href='https://yashpatel5400.github.io/' target='_blank'>Yash Patel</a><sup>1</sup>&emsp;;
    <a href='https://unique-subedi.github.io/' target='_blank'>Unique Subedi</a><sup>1</sup>&emsp;
    <a href='https://www.ambujtewari.com/' target='_blank'>Ambuj Tewari</a><sup>2</sup>&emsp;
</div>

<div align='center'>
Department of Statistics, University of Michigan.
</div>

<p align='center'>
    <sup>1</sup>Equal contributions&emsp;
    <sup>2</sup>Senior investigator
</p>
<div align='center'>
    <a href='https://arxiv.org/abs/2505.18288'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
</div>

## ⚒️ Automatic Pipeline
Linear operator learning for the time-dependent Schrodinger equation.

The following potentials are currently supported
- Free (`free`)
- Harmonic Oscillator (`harmonic_oscillator`)
- Barrier (`barrier`)
- Random (`random`)
- Paul Trap (time-varying) (`paul_trap`)
- Shaken Lattice (`shaken_lattice`)
- Gaussian Pulse (`gaussian_pulse`)
- Coulomb (spherical) (`coulomb`)
- Dipole (spherical) (`dipole`)

The following estimators are currently supported
- New linear estimator (`linear`)
- FNO (`fno`)
- DeepONet (`onet`)

To run the pipeline on a given potential with a given estimator, run the following command (from the supported list above). You can
also do `all` for the estimator to run all the supported estimators. This will generate results in the `models/` and `results` directories,
respectively saving the trained estimators (other than the linear estimator) and the relative errors on the test dataset:
```
python main.py --potential free --estimator all --noise_sigma 1e-4 1e-3 1e-2 --n_jobs 4
```

To then generate the final LaTex results table, run:
```
python eval.py
```

## ⚖️ Disclaimer
This project is intended for academic research, and we explicitly disclaim any responsibility for user-generated content. Users are solely liable for their actions while using the generative model. The project contributors have no legal affiliation with, nor accountability for, users' behaviors. It is imperative to use the generative model responsibly, adhering to both ethical and legal standards.

## &#x1F4D2; Citation

If you find our work useful for your research, please consider citing the paper :

```
@article{patel2025operator,
  title={Operator Learning for Schr$\backslash$"$\{$o$\}$ dinger Equation: Unitarity, Error Bounds, and Time Generalization},
  author={Patel, Yash and Subedi, Unique and Tewari, Ambuj},
  journal={arXiv preprint arXiv:2505.18288},
  year={2025}
}
```