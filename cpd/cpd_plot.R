#
# This file was automatically generated.
#

# changepoint
cp=%(cp)d
# total time -- add 1 for the nc event 
T=%(T)d+1
# sampling points -- add 1 for the no change event 
points=c(%(points)s)+1
# estimated changepoint positions
cp_est=c(%(cp_est)s)
# number of streams
d=%(d)d
# affected subset of streams
subset=which(c(%(subset)s)==1)
# marginal probabilites of streams affected
subset_est=c(%(subset_est)s)
# posterior probabilites of changepoints
tm_data=c(%(tm_data)s)
# title
title='%(title)s'
# statistic
stat='%(stat)s'

# create PDF-file
pdf(file='%(file)s', height=length(points)*1.5, width=8)
col_marker='lightpink'

# bottom, left, up, right
par(mar=c(2, 2.5, 1.75, 0.5), oma=c(0,0,3,0))
layout(matrix(1:(2*length(points)), length(points), 2, byrow = TRUE), widths=c(T,d+T/10))

s_0=1; k_0=1

# colors
cp_colors=c('darkgoldenrod1','darkgoldenrod4')
if (stat=='FSum') cp_colors=c('cornsilk1','cornsilk4')
if (stat=='FMultiple') cp_colors=c('lightsteelblue1','lightsteelblue4')
if (stat=='Bayesian') cp_colors=c('darkorchid1','darkorchid4')
cp_colors=c(cp_colors,col_marker)

# constants
cp_y_max=1.0 #min(round(max(tm_data),1)+0.1,1)
st_names = 1:d
st_colors=c('seagreen1',col_marker)
st_y_max=1.0

for (i in 1:length(points)) {
  
  # changepoint
  cp_names  = c(1:(points[i]-1), 'nc', rep('',T-points[i]))
  cp_mp     = tm_data[s_0:(s_0+points[i])]; s_0=s_0+points[i]
  cp_values = c(cp_mp[1:points[i]-1],rep(0,T-points[i]+1))
  cp_ia     = rep(0, T); cp_ia[points[i]]=cp_mp[points[i]]
  cp_ib     = rep(0, T)
  if (points[i]>cp) cp_ib[cp]=cp_y_max*1.05-cp_values[cp]-cp_ia[cp]
  cp_bplot  = t(array(c(cp_values, cp_ia, cp_ib), c(T,3)))

  # streams
  st_mp=subset_est[k_0:(k_0+d-1)]; k_0=k_0+d
  st_ia=rep(0,d)
  if (points[i]>cp) st_ia[subset]=st_y_max*1.05-st_mp[subset]
  st_bplot = t(array(c(st_mp, st_ia), c(d,2)))
  
  # plot
  barplot(cp_bplot, ylim=c(0, cp_y_max), names=cp_names, las=2, cex.names=0.5, cex.axis=0.75, axes=TRUE, col=cp_colors, space=0, xaxs='i',xlim=c(0, T))
  if (i==1) title(main='the pink bar indicates the true changepoint, the red line indicates the estimate at that time, "nc" is the likelihood of no change', line=1, cex.main=0.75, font.main=1)
  if (cp_est[i]<points[i]) abline(v=cp_est[i]-0.5, col='red', lwd=1+15/T)
  barplot(st_bplot, ylim=c(0, st_y_max), names=st_names, las=2, cex.names=0.5, cex.axis=0.75, axes=TRUE, col=st_colors, space=0, xaxs='i',xlim=c(0, d))
  if (i==1) title(main='affected streams', line=1, cex.main=0.75, font.main=1)
  
}
mtext(title, line=1, cex=0.5, outer=TRUE)
dev.off()