setwd("data")

scale <- function(x) {
  x / x[which.max(abs(x))]
}

psiFilenames <- list.files(pattern="_psi.txt$")
psis <- lapply(psiFilenames, function(f) { read.delim(f, header=TRUE) })
names(psis) <- sub("_psi.txt", "", psiFilenames)

pdf("psiplots.pdf", useDingbats=FALSE)
plot(psis$none$j, scale(psis$none$psi_2), type="l", col="black", ylim=c(-1,1))
lines(psis[["0.5"]]$j, scale(psis[["0.5"]]$psi_2), col="blue")
lines(psis[["1.0"]]$j, scale(psis[["1.0"]]$psi_2), col="cyan")
lines(psis[["2.0"]]$j, scale(psis[["2.0"]]$psi_2), col="green")
lines(psis[["4.0"]]$j, scale(psis[["4.0"]]$psi_2), col="red")
lines(psis[["8.0"]]$j, scale(psis[["8.0"]]$psi_2), col="magenta")

plot(psis$none$j, scale(psis$none$psi_2), type="l", col="black", ylim=c(-1,1))
lines(psis[["8.0"]]$j, scale(psis[["8.0"]]$psi_2), col="blue")
lines(psis[["16.0"]]$j, scale(psis[["16.0"]]$psi_2), col="cyan")
lines(psis[["32.0"]]$j, scale(psis[["32.0"]]$psi_2), col="green")
lines(psis[["64.0"]]$j, scale(psis[["64.0"]]$psi_2), col="red")
lines(psis[["128.0"]]$j, scale(psis[["128.0"]]$psi_2), col="magenta")

plot(psis$none$j, scale(psis$none$psi_2), type="l", col="black", ylim=c(-1,1))
lines(psis[["128.0"]]$j, scale(psis[["128.0"]]$psi_2), col="blue")
lines(psis[["256.0"]]$j, scale(psis[["256.0"]]$psi_2), col="cyan")
lines(psis[["512.0"]]$j, scale(psis[["512.0"]]$psi_2), col="green")
lines(psis[["1024.0"]]$j, scale(psis[["1024.0"]]$psi_2), col="red")

plot(psis[["512.0"]]$j, scale(psis[["512.0"]]$psi_2), type="l", col="black", ylim=c(-1,1))
lines(psis[["512.0"]]$j, scale(psis[["512.0"]]$psi_3), col="red")
lines(psis[["512.0"]]$j, scale(psis[["512.0"]]$psi_4), col="blue")
lines(psis[["512.0"]]$j, scale(psis[["512.0"]]$psi_5), col="magenta")

dev.off()
