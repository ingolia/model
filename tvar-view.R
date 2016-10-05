# psi0 <- read.delim("psi-v0-t.txt", header=FALSE, row.names=1)
psi1 <- read.delim("psi-v1-t.txt", header=FALSE, row.names=1)

#yrange <- c(0, 1.1*max(psi0, psi1))
yrange <- c(0, 1.1*max(psi1, na.rm=TRUE))
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

pdf ("steady.pdf", useDingbats=FALSE)
eig0 <- read.delim("eig-v0-abs2.txt", header=FALSE, row.names=1)
eig1 <- read.delim("eig-v1-abs2.txt", header=FALSE, row.names=1)

plot(t(eig0)[,1], type="l", col="black", ylim=c(0, 1.1*max(eig0, eig1)))
lines(t(eig0)[,2], col="blue")
lines(t(eig0)[,3], col="cyan")
lines(t(eig1)[,1], col="red")
lines(t(eig1)[,2], col="magenta")

dev.off()

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
