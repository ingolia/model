setwd("data")

scale <- function(x) {
  x / x[which.max(abs(x))]
}

pdf("psiplots.pdf", useDingbats=FALSE)
none <- read.delim("none_psi.txt", header=TRUE)
plot(none$j, scale(none$psi_2), pch=21, cex=0.33, ylim=c(-1,1))
lines(none$j, sin(none$j * pi / (max(none$j))), col="red")

half <- read.delim("half_psi.txt", header=TRUE)
plot(half$j, scale(half$psi_2), pch=21, cex=0.33, ylim=c(-1,1))
lines(half$j, sin(half$j * pi / (max(half$j))), col="blue")
lines(half$j, sin(half$j * pi / (0.5 * max(half$j))), col="red")

plot(half$j, scale(half$psi_2), type="l", col="black", ylim=c(-1,1))
lines(half$j, scale(half$psi_3), col="blue")
lines(half$j, scale(half$psi_4), col="cyan")
lines(half$j, scale(half$psi_5), col="green")
lines(half$j, scale(half$psi_6), col="red")
lines(half$j, scale(half$psi_7), col="magenta")
dev.off()
