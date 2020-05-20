echo \<script\> skew 
  
skew=${1}

for i in `seq 1 50`;
do
       echo $i;
	python LR_ppi_cov.py features/krogan_ppis_no3merfeats.csv features/cov2_human_allnegs_no3mer_good.csv ${skew} > models/100trials_LR/cv_trial${i}_skew${skew}.out
done

