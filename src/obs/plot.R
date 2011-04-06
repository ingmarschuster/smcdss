##
## Reads the results from the result.csv-file and plots estimated densities.
##

# constants
colors   = c('CE'='olivedrab4', 'SA'='orangered4', 'SMC'='green')
linetype = c('CE'=1,            'SA'=2,            'SMC'=3)

# read data
data=read.csv('%(resultfile)s')

# construct list of computed problems
problems=unique(data$PROBLEM)

pdf(file='%(pdffile)s', height=8, width=8)

for (problem in problems) {
	
	probdata=subset(data[c('OBJ','BEST_OBJ','ALGO')], data['PROBLEM']==problem)
	
	# determine range
	min_x = min(probdata$OBJ)
	max_x = max(probdata$BEST_OBJ)
	
	# construct list of algorithms
	algos = as.matrix(unique(probdata$ALGO))
	
	# construct kernel density estimates
	density_plot=list(); max_y=list()	
	for(algo in algos) {
		algodata=subset(probdata[c('OBJ','BEST_OBJ')], probdata['ALGO'] == algo)
		density_plot[[algo]] = density(algodata$OBJ, to=max_x)
		max_y[algo] = max(density_plot[[algo]]$y)
	}
	
	# determine total maximum y
	max_y = max(unlist(max_y))
	
	# plot
	plot(x=c(min_x, max_x), y=c(0,max_y), type="n", family='serif', cex.axis=1, cex.lab=1, xlab='objective', ylab='')
	for(algo in algos) {
		lines(density_plot[[algo]]$x, density_plot[[algo]]$y, type='l', lty=linetype[algo], col=colors[algo])
	}
	
	# add vertical line at best known objective value
	abline(v = algodata$BEST_OBJ[1], col = 'royalblue')
	
	# add legend
	legend(x='topleft', inset=0.1, legend=c('Cross-Entropy','Simulated Annealing','Sequential Monte Carlo'), cex=1, col=colors, lty=linetype)
	
}
dev.off()