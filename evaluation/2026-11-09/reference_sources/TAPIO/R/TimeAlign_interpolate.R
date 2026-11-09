
TimeAlign_interpolate = function(Longdat2_wide){
require(zoo)

y = Longdat2_wide[,4:ncol(Longdat2_wide)]
old_t = Longdat2_wide$time
#we first transfer the observations to the same times!
#quantile(Longdat$Dat$obs, p = seq(0,1,length=11))
interPolateTS = function(y, old_t, new_t = 1:10){
  library(zoo)
  y_ts = zoo(y, old_t)
  z_ts = zoo(order.by = new_t*1.0)
  # Merge series into one object
  z <- merge(y_ts,z_ts)
  z_int = na.approx(z, rule = 2)
  return(window(z_int, 1:10))
}

ids = unique(Longdat2_wide$subject)
for (i in ids) {
  jj = which(Longdat2_wide$subject == i)
  tmp = as.data.frame(interPolateTS(y[jj,],old_t[jj]))
  tmp$id = i
  tmp$time = 1:10
  tmp$label = unique(Longdat2_wide$cluster[jj])
  if (i == 1) {
    y_int = tmp
  } else {
    y_int = rbind.data.frame(y_int, tmp)
  }
}

y_int2 = y_int[,c(6:8,1:5)]
NAMES = colnames(y_int2)
NAMES = c("subject","time","cluster",NAMES[-(1:3)])
colnames(y_int2) = NAMES
return(y_int2)
}