# psi0 <- read.delim("psi-v0-t.txt", header=FALSE, row.names=1)
psi1 <- read.delim("psi-v1-t.txt", header=FALSE, row.names=1)

#yrange <- c(0, 1.1*max(psi0, psi1))
yrange <- c(0, 1.1*max(psi1))
#plot(t(psi0)[,1], type="l", col="black", ylim=yrange)

plot(t(psi1)[,1], type="l", col="black", ylim=yrange)
lines(t(psi1)[,64], col="blue")
lines(t(psi1)[,128], col="cyan")
lines(t(psi1)[,192], col="green")
lines(t(psi1)[,256], col="red")
lines(t(psi1)[,512], col="magenta")

plot(psi1$V64, type="l", col="black", ylim=yrange)
lines(psi1$V32, col="blue")
lines(psi1$V96, col="red")

pdf("heatmap.pdf", useDingbats=FALSE)
image(z=as.matrix(psi1), zlim=yrange, col=topo.colors(32))

dev.off()
pdf("animation.pdf", useDingbats=FALSE)
for (i in seq(9,nrow(psi1),8)) {
  plot(t(psi1)[,i], type="l", ylim=yrange, lwd=2)
  for (j in seq(i-1,i-3,-1)) {
    lines(t(psi1)[,j])
  }
  for (j in seq(i-4,i-8,-1)) {
    lines(t(psi1)[,j], col="grey")
  }
  title(main=sprintf("%d to %d", i, i-8))
}
dev.off()
