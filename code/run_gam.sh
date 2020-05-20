echo \<script\> skew interac
  
skew=${1}
interac=${2}

for i in `seq 1 50`;
do
       echo $i;
	python GAM_ppi_cov.py features/krogan_ppis_no3merfeats.csv features/cov2_human_allnegs_no3mer_good.csv ${skew} ${interac} > models/100trials/cv_int${interac}_trial${i}_skew${skew}.out
done

