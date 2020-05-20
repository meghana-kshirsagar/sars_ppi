echo \<script\> skew 
  
skew=${1}

for i in `seq 1 25`;
do
       echo $i;
	python LR_ppi_cov.py features/krogan_humanprot_no3merfeats.csv features/all_humanprots_nokrogan_no3merfeats.csv ${skew} > models/hprots_100trials_LR/cv_trial${i}_skew${skew}.out

done

