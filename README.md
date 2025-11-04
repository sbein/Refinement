# Refinement
To load the needed packages create a conda environment using the [yml file](refinement_env.yml) or use [LCG](https://lcgdocs.web.cern.ch/lcgdocs/lcgreleases/introduction/) release [106_cuda](https://lcginfo.cern.ch/release/106_cuda/), e.g.,
```
source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh
```

This is the current sequence I use to do a training and produce plots on the website:

```
screen
condor_submit -i big.submit
source /afs/desy.de/user/b/beinsam/.bash_profile
cd /data/dust/user/beinsam/FastSim/Refinement/Regress
source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh
mkdir 18April2025lowlr && cp trainRegression_Jet.py my*.py 18April2025lowlr/
python trainRegression_Jet.py 18April2025lowlr 2>&1 | tee traininglog_regressionJethighlr.txt
cd plotting
python plotLearningCurves.py 18April2025lowlr &
python plotRegression1D.py 18April2025lowlr
python cropPNGlinlog.py "figs*/reg*.png"
find figs*/ -type f -name 'reg*.png' \! -name '*log.png*' -exec rm {} + && rm fig*/*{mmdfixsigma_output,mse_input_output}*
python plotRegressionCorrelationFactors.py 18April2025lowlr
rm -rf /afs/desy.de/user/b/beinsam/www/FastSim/Refinement/Jets/figs18April2025lowlr*
cp -r figs18April2025lowlr /afs/desy.de/user/b/beinsam/www/FastSim/Refinement/Jets
python /afs/desy.de/user/b/beinsam/www/templates/dir_indexer.py /afs/desy.de/user/b/beinsam/www/FastSim/Refinement/Jets -r -t /afs/desy.de/user/b/beinsam/www/templates/default.html && python /afs/desy.de/user/b/beinsam/www/templates/bigindexer.py /afs/desy.de/user/b/beinsam/www/FastSim/Refinement/Jets/
cp ../18April2025lowlr/* /afs/desy.de/user/b/beinsam/www/FastSim/Refinement/Jets/18April2025lowlr/
```
