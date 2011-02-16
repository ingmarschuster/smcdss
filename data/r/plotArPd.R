##
## Reads the tab-separated csv-files ar_pm, ar_lrm, pd_pm and pd_lrm
## and creates two plots. If out is TRUE a pdf-file is created in the
## same folder.
##

plotArPd = function(path='../data/testruns/arpd_analysis',
		colors = c('black','deepskyblue'), types=c('ar','pd'), out=FALSE) {
	
	for (type in types) {
		
		# set position of legend
		if (type == 'ar') pos='topright'
		if (type == 'pd') pos='bottomleft'
		
		# plot parameters
		files = c(paste(type,'pm.csv',sep='_'),paste(type,'lrm.csv',sep='_'))
		d = length(files)
		plotchar = c(15,19)
		linetype = c(1:d)
		
		# open pdf device
		if (out) pdf(file=paste(path, paste(type,'analysis.pdf',sep='_'), sep='/'), height=6, width=10)
		
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
				cex=2, col=colors, lty=linetype)
		par(family='serif')
		
		# turn device off
		if (out) dev.off()
	}
}

plotArPd(colors = c('black','chocolate4'), out=TRUE)