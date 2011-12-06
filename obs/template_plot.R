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
primal_bound = %(primal_bound)d

# number of exact bars
exact=%(exact)s
# number of aggregated bars
bars=%(bars)s
# number of aggregated bars
ntotal=%(n)d

# read data
data=read.csv('%(resultfile)s')
d=length(names(data)[substr(names(data), start=1, stop=1)=='S'])
title='%(title)s'
if (title!='') title=paste(title, d)

if (type=='algo')  probdata=data[c('OBJ','ALGO')]
if (type=='model') probdata=data[c('OBJ','MODEL')]

# construct list of algorithms
primal_bound= max(max(probdata$OBJ), primal_bound)
primal_bound = primal_bound[1:ntotal]

breaks=unique(probdata$OBJ)
if (length(breaks) == 1) breaks[2]=0
if (length(breaks)<=exact) {
	names=breaks[1:length(breaks)-1]
} else {
	aggre=min(bars-exact,length(breaks)-exact)
	breaks=breaks[order(breaks, decreasing=TRUE)]
	aggre=seq(max(breaks[exact:length(breaks)]),min(breaks),length.out=aggre)[2:aggre]
	breaks=c(breaks[1:exact],aggre)
	names=breaks[1:exact]
	for (i in (exact+1):(exact+length(aggre)-1)) names[i]=paste(floor(breaks[i]),'<')
}

# PDF
pdf(file='%(pdffile)s', height=4, width=2+0.175*length(breaks))
par(mar=c(%(mar)s), family='serif')

# construct kernel density estimates
hist_single=list()
if (type=='algo') {
	for(algo in algos) {
		# select results generated from algo
		algodata=subset(probdata['OBJ'], probdata['ALGO'] == algo)
		if (dim(algodata)[1] > 0) {
			# select number of test runs
			n[algo]=dim(algodata)[1]
			m[algo]=length(unique(algodata$OBJ))
		}
		# select inverted counts
		hist_single[[algo]]=hist(algodata$OBJ, plot=FALSE, breaks=breaks)$counts[(length(breaks)-1):1]
	}
}
if (type=='model') {
	for(model in models) {
		# select results generated from algo
		modeldata=subset(probdata['OBJ'], probdata['MODEL'] == model)
		if (dim(algodata)[1] > 0) {
			# select number of test runs
			modeldata=modeldata[1:min(dim(modeldata)[1], ntotal),]
			n[model]=dim(modeldata)[1]
			m[model]=length(unique(modeldata$OBJ))
		}
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
	legend(x='topright', inset=0.05, cex=1, fill=colors, legend=c(
					paste('Simulated Annealing [', m['SA'], '/', n['SA'], ']', sep=''),
					paste('Cross Entropy [', m['CE'], '/', n['CE'], ']', sep=''),
					paste('Sequential Monte Carlo [', m['SMC'], '/', n['SMC'], ']', sep='')))
}
if (type=='model') {
	legend(x='topright', inset=0.05, cex=1, fill=colors, legend=c(
					paste('Product model [', m['product'], '/', n['product'], ']', sep=''),
					paste('Logistic conditional model [', m['logistic'], '/', n['logistic'], ']', sep=''),
					paste('Gaussian copula model [', m['gaussian'], '/', n['gaussian'], ']', sep='')))
}
dev.off()
