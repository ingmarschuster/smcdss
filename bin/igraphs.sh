#!/bin/sh
ARTICLE=$HOME/Documents/Latex/Papers/smc_on_lbs/springer
DATA=$HOME/Documents/Python/smcdss/data

TOY=$DATA/r/pdf
ARPD=$DATA/testruns/arpd_analysis
TRUN=$DATA/testruns

cp $TOY/*.pdf $ARTICLE
cp $ARPD/*.pdf $ARTICLE
cp $TRUN/boston_smc/eval.pdf $ARTICLE/boston_smc.pdf
cp $TRUN/boston_mcmc/eval.pdf $ARTICLE/boston_amcmc.pdf
cp $TRUN/boston_mcmc_sym/eval.pdf $ARTICLE/boston_mcmc.pdf
cp $TRUN/concrete_smc/eval.pdf $ARTICLE/concrete_smc.pdf
cp $TRUN/concrete_mcmc/eval.pdf $ARTICLE/concrete_amcmc.pdf
cp $TRUN/concrete_mcmc_sym/eval.pdf $ARTICLE/concrete_mcmc.pdf

FILES="
ar_analysis
pd_analysis
function
product
logregr
boston_smc
boston_amcmc
boston_mcmc
concrete_smc
concrete_amcmc
concrete_mcmc"
for f in $FILES
do
	echo $f
	pdftops -eps $f.pdf
done

