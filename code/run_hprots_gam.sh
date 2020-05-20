echo \<script\> skew interac
  
skew=${1}
interac=${2}

for i in `seq 1 25`;
do
       echo $i;
	python GAM_ppi_cov.py features/krogan_humanprot_no3merfeats.csv features/all_humanprots_nokrogan_no3merfeats.csv ${skew} ${interac} > models/hprot_100trials/cv_int${interac}_trial${i}_skew${skew}.out
done

