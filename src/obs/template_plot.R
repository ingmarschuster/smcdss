##
## Reads the results from the result.csv-file and plots estimated densities.
##

# constants
legend = %(legend)s
counts = %(counts)s
type   = '%(type)s'
algos  = c('CE', 'SMC', 'SMCL', 'SA', 'RLS')
models = c('product', 'logistic', 'gaussian')
colors = c(%(colors)s)
n      = c('SA'=0, 'CE'=0, 'SMC'=0, 'SMCL'=0, 'RLS'=0, 'gaussian'=0, 'logistic'=0, 'product'=0)
m      = c('SA'=0, 'CE'=0, 'SMC'=0, 'SMCL'=0, 'RLS'=0, 'gaussian'=0, 'logistic'=0, 'product'=0)
primal_bound = %(primal_bound)s

shaded=NULL
for (i in 1:length(colors)) {
	shaded[i]=rgb(t(col2rgb(colors[i])*0.2/100))
}

# number of exact bars
exact=%(exact)s
# number of aggregated bars
bars=%(bars)s
# number of aggregated bars
ntotal=%(n)d
# digits of percentage
digits=3

# read data
data=read.csv('%(resultfile)s')
d=length(names(data)[substr(names(data), start=1, stop=1)=='S'])
title='%(title)s'
if (title!='') title=paste(title, d)

if (type=='algo') probdata=data[c('OBJ','ALGO')]
if (type=='model') probdata=data[c('OBJ','MODEL')]

# compute results relative to best known value and worst run
max_v=max(max(probdata$OBJ), primal_bound)
min_v=min(probdata$OBJ)
probdata$OBJ=(probdata$OBJ-min_v)/(max_v-min_v)

# setup breaks
breaks=unique(probdata$OBJ)
exact=min(exact,length(breaks))
if (length(breaks) < bars) breaks=c(breaks,rep(-1,bars-length(breaks)))

# aggregate breaks (if more breaks than bars) and create names
if (length(breaks) == 1) breaks[2]=0
if (length(breaks)<=exact+1) {
	names=format(breaks[1:length(breaks)-1],digits=digits)
} else {
	aggre=min(bars-exact,length(breaks)-exact)
	breaks=breaks[order(breaks, decreasing=TRUE)]
	aggre=seq(max(breaks[exact:length(breaks)]),min(breaks),length.out=aggre-1)
	breaks=c(breaks[1:exact],aggre)
	names=format(round(breaks[1:exact],digits=digits))
	for (i in (exact+1):(exact+length(aggre)-1)) {
    print(breaks[i+1])
		if (breaks[i+1]<=0) names[i]='0.000<'
		else names[i]=paste(format(round(breaks[i+1], digits=digits)),'<',sep='')
	}
}

# PDF
pdf(file='%(pdffile)s', height=6, width=2+0.5*length(breaks))
par(mar=c(%(mar)s), family='serif')

# construct kernel density estimates
hist_single=list()
if (type=='algo') {
	for(algo in algos) {
		# select results generated from algo
		algodata=as.list(subset(probdata['OBJ'], probdata['ALGO'] == algo))
		if (length(algodata$OBJ) > 0) {
			# select number of test runs
			algodata=algodata$OBJ[1:min(length(algodata$OBJ),ntotal)]
			n[algo]=length(algodata)
			m[algo]=length(unique(algodata))

			# select inverted counts
			hist_single[[algo]]=hist(algodata, plot=FALSE, breaks=breaks)$counts[(length(breaks)-1):1]
		}
	}
}
if (type=='model') {
	for(model in models) {
		# select results generated from algo
		modeldata=as.list(subset(probdata['OBJ'], probdata['MODEL'] == model))
		if (length(modeldata$OBJ) > 0) {
			# select number of test runs
			modeldata=modeldata$OBJ[1:min(length(modeldata$OBJ),ntotal)]
			n[model]=length(modeldata)
			m[model]=length(unique(modeldata))
		}
		# select inverted counts
		hist_single[[model]]=hist(modeldata, plot=FALSE, breaks=breaks)$counts[(length(breaks)-1):1]
	}
}

# plot multi-histogram
hist_multi=NULL
for (i in 1:length(hist_single)) {
  hist_multi=rbind(hist_multi, hist_single[[i]])
}

# plot bars and axis
barplot(hist_multi, names=names, axis.lty=0, tck=0.01, las=2,
		main=title, cex.names=2.5, cex.axis=2.5, border=FALSE, col=colors, mgp=c(3, 0.25, 0),
		space=0.1, xaxs='i', xlim=c(-.25, length(breaks)+2))

# plot vertical lines and border
barplot(hist_multi, axes=FALSE, angle=c(30,60,90,-60,-30), border=shaded,
		col=shaded, density=15, space=0.1, xlim=c(-.25, length(breaks)+2), add=TRUE)

# add legend
if (counts) {
	if (type=='algo') {
		vlegend=c(paste('Cross-entropy',' [', m['CE'], '/', n['CE'], ']', sep=''),
				  paste('SMC parametric',' [', m['SMC'], '/', n['SMC'], ']', sep=''),
				  paste('SMC symmetric',' [', m['SMCL'], '/', n['SMCL'], ']', sep=''),
				  paste('Simulated annealing',' [', m['SA'], '/', n['SA'], ']', sep=''),
				  paste('1-opt local search',' [', m['RLS'], '/', n['RLS'], ']', sep=''))
	} else {
		vlegend=c(paste('Product model',' [', m['product'], '/', n['product'], ']', sep=''),
				  paste('Logistic conditionals model',' [', m['logistic'], '/', n['logistic'], ']', sep=''),
				  paste('Gaussian copula model',' [', m['gaussian'], '/', n['gaussian'], ']', sep=''))
	}
} else {
	if (type=='algo') {
		vlegend=c(paste('Cross-entropy', sep=''),
				  paste('SMC parametric', sep=''),
				  paste('SMC symmetric', sep=''),
				  paste('Simulated annealing', sep=''),
				  paste('1-opt local search', sep=''))
	} else {
		vlegend=c(paste('Product model', sep=''),
				  paste('Logistic conditionals model', sep=''),
				  paste('Gaussian copula model', sep=''))
	}
}
if (legend) {
  inv=length(colors):1
  legend(x='topright', inset=0.05, cex=2.5,
         fill=colors[inv], legend=vlegend[inv])
  legend(x='topright', inset=0.05, cex=2.5, border=shaded[inv],
         fill=shaded[inv], angle=c(30,60,90,-60,-30)[inv], density=15, legend=vlegend[inv])
}
dev.off()
