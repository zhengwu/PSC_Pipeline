This PSC pipeline is developed with the help of [scilpy](https://github.com/scilus/scilpy) under python 2.7. So to use this PSC pipeline, an old python 2 scilpy should be installed first. Here I breifly describe how to install and use PSC.



-  Step 1. Install a python2 version of scilpy through an installation of one of its variants `SET`. Instruction can be found [here](https://set-documentation.readthedocs.io/en/latest/setup/python.html#python-2-7-for-set-dev)

- Step 2.  To setup the PSC pipeline,  clone the PSC pipeline: git clone https://github.com/zhengwu/PSC_Pipeline.git and change the environmental variables in your
  `.bashrc` or `.bash_profile`(in linux of Mac OS):

  ```
  export PATH="/home/username/Software/PSC_Pipeline/scripts:$PATH"
  export PYTHONPATH="/home/username/Software/PSC_Pipeline:$PYTHONPATH"
  ```

  Now you should be able to run commands in the `scripts` folder.

- Step 3. Check the pipeline scripts inside the 'abcd_dwi_preprocessing' folder for precessing dwi data from ABCD/UK Biobank. 

Note that - 1. PSC relies on MRtrix, FSL and freesurfer, so make sure these have been installed first. 2. you might need to change some dir in the scripts so that your system can find correct files. 
