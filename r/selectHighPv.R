##
## Selects columns that have a -log10(p-value) greater than a given treshold min_val
## for the t-statistic of H0: b1=0 in a univariate linear regression model.
##
## Then modify_cols is called to create a reduced data set for further processing.
##

source('modify_cols.R')

selectHighPv = function(file, y=1, X=2, out_file='', min_val=2.0, intercept=TRUE, dec=".", sep=',', head=TRUE) {
	
	# read csv file
	in_data = read.csv(file, sep=sep, dec=dec, head=head);
	
	# determine covariate columns
	if (length(X) == 1) X = seq(X, dim(in_data)[2]);
	
	# get covariate columns
	covariates = colnames(in_data[X]);
	
	Z = c(); i = 0;
	
	# loop over all covariates
	for(col in covariates) {
		
		# compute -log10(p-value)
		val = -log10(summary(lm(in_data[,y]~in_data[,col]))$coefficients[2,'Pr(>|t|)']);
		
		# append covariate if the criterion is met
		if (val >  min_val) {	
			i = i + 1;
			Z[i] = col;
		}
		
	}
	# print ratio
	print(paste(length(Z), '/', length(X)))
	rm(in_data)
	
	modifyCols(file, y=y, X=Z, out_file=out_file, intercept=TRUE,
			add_interactions=FALSE, add_polynomials=FALSE, dec=dec, sep=sep, head=head)
	
}