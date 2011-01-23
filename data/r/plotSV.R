##
## The function computes and plots the singular value decomposition for each
## each dataset given in the vector files. The function expects standard
## CSV-files, that is sep=',' and dec='.'.
##
## If out is TRUE, a pdf-file is created.
##

plot_sv = function(files, path='../data/dataset', out=FALSE) {
	
	d = length(files)
	colors = gray(1:(d+1)/(d+5))
	linetype = c(1:d)
	if (out) pdf(paper="a4r", file='pdf/sv_analysis.pdf')
	plot(x=c(0 ,0.2), y=c(0 ,1), type="n", xlab="singular values",
			ylab="normalized value", main="Singular value analysis")
	
	for (i in 1:d) {
		data = read.csv(file=paste(path,files[i],sep='/'))
		data = data[2:dim(data)[2]]
		y =svd(data)$d
		y = y / max(y)
		x = seq(from=0, to=1, length.out=length(y))
		lines(x=x, y=y, lty=linetype[i], col=colors[i])
	}
	legend(x=0.05, y=1, legend=files, cex=0.8, col=colors, lty=linetype, title="datasets")
	if (out) dev.off()
}