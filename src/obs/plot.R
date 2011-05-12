##
## Reads the results from the result.csv-file and plots estimated densities.
##

# constants
type   ='%(type)s'
algos  = c('SA', 'CE', 'SMC')
models = c('product', 'logistic', 'gaussian')
colors = c(%(colors)s)
n      = c('SA'=0, 'CE'=0, 'SMC'=0, 'gaussian'=0, 'logistic'=0, 'product'=0)
m      = c('SA'=0, 'CE'=0, 'SMC'=0, 'gaussian'=0, 'logistic'=0, 'product'=0)

# number of exact bars
exact=10
# number of aggregated bars
aggre=10

# read data
data=read.csv('%(resultfile)s')
d=length(names(data)[substr(names(data), start=1, stop=1)=='S'])
title='%(title)s'
if (title!='') title=paste(title, d)

# construct list of computed problems
problems=unique(data$PROBLEM)

for (problem in problems) {
	
	if (type=='algo')  probdata=subset(data[c('OBJ','BEST_OBJ','ALGO')], data['PROBLEM']==problem)
	if (type=='model') probdata=subset(data[c('OBJ','BEST_OBJ','MODEL')], data['PROBLEM']==problem)

	# construct list of algorithms
	best_obj = probdata$BEST_OBJ[1]
	if (best_obj=='-Inf') best_obj=max(probdata$OBJ)
	
	# determine range
	#min_x = min(probdata$OBJ)
	#max_x = max(best_obj)
	
	breaks=unique(probdata$OBJ)
	if (length(breaks)<=exact) {
		names=breaks[1:length(breaks)-1]
	} else {
		aggre=min(aggre,length(breaks)-exact)
		breaks=breaks[order(breaks, decreasing=TRUE)]
		breaks=c(breaks[1:exact],seq(max(breaks[exact:length(breaks)]),min(breaks),length.out=aggre))
		names=breaks[1:exact]
		for (i in exact:(exact+aggre-1)) names[i]=paste(floor(breaks[i]),'<')
	}
	
	pdf(file='%(pdffile)s', height=4, width=2+0.175*length(breaks))
	par(mar=c(%(mar)s), family='serif')
	
	# construct kernel density estimates
	hist_single=list()
	if (type=='algo') {
		for(algo in algos) {
			# select results generated from algo
			algodata=subset(probdata[c('OBJ','BEST_OBJ')], probdata['ALGO'] == algo)
			# select number of test runs
			n[algo]=dim(algodata)[1]
	  m[algo]=length(unique(algodata$OBJ))
			# select inverted counts
			hist_single[[algo]]=hist(algodata$OBJ, plot=FALSE, breaks=breaks)$counts[(length(breaks)-1):1]
		}
	}
	if (type=='model') {
		for(model in models) {
			# select results generated from algo
			modeldata=subset(probdata[c('OBJ','BEST_OBJ')], probdata['MODEL'] == model)
			# select number of test runs
			n[model]=dim(modeldata)[1]
	  		m[model]=length(unique(modeldata$OBJ))
			# select inverted counts
			hist_single[[model]]=hist(modeldata$OBJ, plot=FALSE, breaks=breaks)$counts[(length(breaks)-1):1]
		}
	}
	
	# plot multi-histogram
	hist_multi=rbind(hist_single[[1]], hist_single[[2]], hist_single[[3]])
	barplot(hist_multi, names=names, axis.lty=0, tck=0.01, las=2,
			main=title, cex.names=0.5, cex.axis=0.75, col=colors, mgp=c(3, 0.25, 0),
			xaxs='i', xlim=c(-1, length(breaks)*1.2 + 1)) 
	
	# add legend
	if (type=='algo') {
		legend(x='topright', inset=0.1, cex=1, fill=colors, legend=c(
						paste('Simulated Annealing [', n['SA'], ']', sep=''),
						paste('Cross Entropy [', n['CE'], ']', sep=''),
						paste('Sequential Monte Carlo [', n['SMC'], ']', sep='')))
	}
	if (type=='model') {
		legend(x='topright', inset=0.1, cex=1, fill=colors, legend=c(
						paste('Product model [', m['product'], '/', n['product'], ']', sep=''),
						paste('Logistic conditional model [', m['logistic'], '/', n['logistic'], ']', sep=''),
						paste('Gaussian copula model [', m['gaussian'], '/', n['gaussian'], ']', sep='')))
	}
}
dev.off()
