##
## Reads the results from the result.csv-file and plots estimated densities.
##

# constants
algos  = c('SA', 'CE', 'SMC')
colors = c('SA'='firebrick4', 'CE'='lightcyan', 'SMC'='gold')
n      = c('SA'=0, 'CE'=0, 'SMC'=0)
bars   = 0

# read data
data=read.csv('%(resultfile)s')
d=length(names(data)[substr(names(data), start=1, stop=1)=='S'])

# construct list of computed problems
problems=unique(data$PROBLEM)

pdf(file='%(pdffile)s', height=8, width=8)

for (problem in problems) {
	
	probdata=subset(data[c('OBJ','BEST_OBJ','ALGO')], data['PROBLEM']==problem)
	
	# construct list of algorithms
	best_obj = probdata$BEST_OBJ[1]
	if (best_obj=='-Inf') best_obj=max(probdata$OBJ)
	
	# determine range
	min_x = min(probdata$OBJ)
	max_x = max(best_obj)
	
	if (bars==0) {
		breaks=unique(probdata$OBJ)
		breaks=breaks[order(breaks, decreasing=TRUE)]
	} else {
		breaks=as.integer(seq(max_x,min_x, length.out=bars))
	}
	
	# construct kernel density estimates
	hist_single=list()
	for(algo in algos) {
		# select results generated from algo
		algodata=subset(probdata[c('OBJ','BEST_OBJ')], probdata['ALGO'] == algo)
		# select number of test runs
		n[algo]=dim(algodata)[1]
		# select inverted counts
		hist_single[[algo]]=hist(algodata$OBJ, plot=FALSE, breaks=breaks)$counts[(length(breaks)-1):1]
	}
	
	# plot multi-histogram
	hist_multi=rbind(hist_single[[1]], hist_single[[2]], hist_single[[3]])
	barplot(hist_multi, names=breaks[1:(length(breaks)-1)], axis.lty=0, tck=0.01, las=2,
			main=paste('%(title)s', d), cex.names=0.5, cex.axis=0.75, col=colors, mgp=c(3, 0.25, 0)) 
	
	# add legend
	legend(x='topright', inset=0.1, cex=1, fill=colors, legend=c(
					paste('Simulated Annealing [', n['SA'], ']', sep=''),
					paste('Cross Entropy [', n['CE'], ']', sep=''),
					paste('Sequential Monte Carlo [', n['SMC'], ']', sep='')))
	
}
dev.off()