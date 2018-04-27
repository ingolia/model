library("RColorBrewer")

E <- read.csv("ramp-E.csv", header=FALSE)
psis <- read.csv("ramp-psi.csv", header=FALSE)

plot(E$V1, E$V2, type="l", lwd=2)
plot(E$V1, E$V3, ylim=c(7.5, 30), type="l", lwd=2)
lines(E$V1, E$V4, lwd=2)

plotpsis <- function(i=0, estep=8) {
  sisp <- t(psis)
  pal <- brewer.pal(9, "OrRd")
  ecols <- seq(2, by=estep, length.out=8)
  psirows <- (36*i + 2):(36*i + 37)
  plot(sisp[psirows,2], type="l", ylim=c(0,1), lwd=2, col=pal[[2]])
  for (i in (1:length(ecols))) {
    lines(sisp[psirows,ecols[[i]]], col=pal[[i+1]], lwd=2)
  }
}

plotpsis(0, 8)
plotpsis(1, 8)
plotpsis(2, 8)

plotpsicirc <- function(psi) {
  n <- length(psi)
  x <- sin(seq(1:n) * 2 * pi / n)
  y <- cos(seq(1:n) * 2 * pi / n)

  vs <- abs(psi)
  hs <- (pi + Arg(psi)) / (2 * pi)
  
  par(bg="black")
  plot(x, y, pch=20, col=hsv(hs, 1, vs), cex=(100/n), 
       asp=1, axes=FALSE, xlab=NA, ylab=NA,
       xlim=c(-1.2,1.2), ylim=c(-1.2,1.2))
}

plotpsicirc(exp(0+1i * seq(0,12,0.2)) * seq(0,1,0.2/12))

foo <- sin(seq(1,36) * 2*pi/36) * (0+1i)
write.csv(foo, "plotting-test-psi.csv", row.names=FALSE)
bar <- read.csv("plotting-test-psi.csv", colClasses=c("complex"))
plotpsicirc(bar$x)

psifiles <- list.files("evolve/", pattern="psi-[0-9]*.csv", full.names=TRUE)
for (psifile in psifiles) {
  psi <- read.csv(psifile, colClasses=c("complex"), header=FALSE)
  pngfile <- sub(".csv$", ".png", psifile)
  png(pngfile, width=200, height=200)
  plotpsicirc(psi$V1)
#  text(x=0, y=-1.2, labels=psifile, cex=0.5, col="white")
  dev.off()
}

# convert -delay 100 -dispose None ring_six/evolve/*.png evolve.gif