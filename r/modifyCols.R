##
## The function modifies a dataset to be processed to variable selection.
##

modifyCols = function(file, y=1, X=2, out_file="", intercept=TRUE, logarithm=FALSE,
		interactions=FALSE, polynomials=1, dec=".", sep=",", head=TRUE) {
	
	# read csv file
	in_data = read.csv(file, sep=sep, dec=dec, head=head);
	
	# determine covariate columns
	if (length(X) == 1) X = seq(X, dim(in_data)[2]);

	# get covariate columns
	covariates = colnames(in_data[X]);
	
	# set observed variable
	out_data = in_data[y];

	if (intercept) out_data['CONST'] = 1;
	
	# loop over all covariates
	for(col in covariates) {
		
		# add covariate
		out_data[col] = in_data[col]
		
		# add logarithm
		if (logarithm) {

			if (min(out_data[col]) > 0) out_data[paste(col, "LOG", sep=".")] = log(out_data[col])

		}

		# add polynomial
		if (polynomials > 1) {
			for(p in seq(2, polynomials)) {

				# add polynomial column
				poly_col = in_data[col] * in_data[col] ^ (p-1)
				if (!all(poly_col == out_data[col])) out_data[paste(col, paste("POW", p,sep=''), sep='.')] = poly_col
			}
		}	
		
		# add interactions
		if (interactions) {
			for(interaction_col in covariates) {
				if (interaction_col == col) break;
				out_data[paste(col, "x", interaction_col, sep=".")] = in_data[col] * in_data[interaction_col]
			}
		}
	}

	# normalize
	# out_data = sweep(out_data, 2, mean(out_data), "/")
	
	# write csv file
	if (out_file == "") out_file = paste(unlist(strsplit(file, split="\\."))[1], ".out", sep="")
	write.table(out_data, file=out_file, row.names=FALSE, sep=",")
	
	return(out_data)
}

plotArPd = function(path, types=c('ar','pd'), out=FALSE) {
	
	for (type in types) {

		# set position of legend
		if (type == 'ar') pos='topright'
		if (type == 'pd') pos='bottomleft'

		# plot parameters
		files = c(paste(type,'pm.csv',sep='_'),paste(type,'lrm.csv',sep='_'))
		d = length(files)
		plotchar = c(15,19)
		colors = c('black','blue')
		linetype = c(1:d)

		# open pdf device
		if (out) pdf(file=paste(path, paste(type,'analysis.pdf',sep='_'), sep='/'))

		# change outer margins and font type	
		savefont = par(mar=c(2.5, 2.5, 0.05, 0.05), family='serif') 
		plot(x=c(0,1), y=c(0,1), type="n", family='serif', cex.axis=1.5, cex.lab=1.5,
			xlab=expression(paste("advance ",rho)), ylab='')

		for (i in 1:d) {
			y = read.csv(paste(path, files[i], sep='/'), sep='\t', head=FALSE)
			x = seq(from=0, to=1, length.out=length(y))
			lines(x=x, y=y, lty=linetype[i], pch=plotchar[i], type='o', col=colors[i])
		}
		# add legend
		legend(x=pos, inset=0.1, pch=plotchar, legend=c('product model','logistic regression model'),
			cex=1.5, col=colors, lty=linetype)
		par(family='serif')

		# turn device off
		if (out) dev.off()
	}
}

