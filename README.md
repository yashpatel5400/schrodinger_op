# Schrodinger Operator
Linear operator learning for the time-dependent Schrodinger equation.

The following potentials are currently supported
- Free (`free`)
- Harmonic Oscillator (`harmonic_oscillator`)
- Barrier (`barrier`)
- Random (`random`)
- Paul Trap (time-varying) (`paul_trap`)

The following estimators are currently supported
- New linear estimator (`linear`)
- FNO (`fno`)
- DeepONet (`onet`)

To run the pipeline on a given potential with a given estimator, run the following command (from the supported list above). You can
also do `all` for the estimator to run all the supported estimators. This will generate results in the `models/` and `results` directories,
respectively saving the trained estimators (other than the linear estimator) and the relative errors on the test dataset:
```
python main.py --potential [potential] --estimator [estimator]
```

To then generate the final LaTex results table, run:
```
python eval.py
```