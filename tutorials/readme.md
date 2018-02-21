- These are tutorials that take you through most of the SCILPY scripts and dev_scripts.
- The tutorials will also take you from raw data to connectomics and tractometry.

Tutorials assume that you have latest installations of:
- SCILPY
- DIPY
- FSL
- ANTS
- TractQuerier
- MRTRIX 0.2.10 (yes, I still use MRTRIX 0.2.10. There was a change of
spherical harmonics basis between MRTRIX 0.2 and MRTRIX 3 with a sqrt(2)
normalization)
- Matplotlib

- Each tutorial requires input data that will automatically be fetched
- You might want to run scripts line by line to learn from them
- Each tutorial creates output in the output/ directory

- It is also recommended to follow slides at the same time:
  + [SCIL pipeline](https://www.dropbox.com/s/0nv1jl2bwdw8zra/scil_dmri_pipeline.pdf?dl=0)
  + [Diffusion MRI course from IMN530](https://www.dropbox.com/s/3byo281dnieu2mg/IRMd.pdf?dl=0)

Enjoy! Give feedback! and contribute with new tutorials!

SETUP
-----

To make sure that the tutorial scripts run easily, make sure that your
SCILPY/scripts path is included in your PATH.

You would normally do something like

```
export PATH=$PATH:scilpy_path/scripts
```
