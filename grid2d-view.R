plotState <- function(state) {
  psi <- read.delim(sprintf("grid2ddata/eig-%s-abs2.txt", state), header=FALSE, row.names=NULL)	  
  yrange <- c(0, 1.1*max(psi, na.rm=TRUE))
  
  pdf(sprintf("grid2ddata/eig-%s.pdf", state), useDingbats=FALSE)

  image(z=as.matrix(psi), zlim=yrange, col=topo.colors(64))

  dev.off()
}

eigfiles <- list.files("grid2ddata", "eig-.*-abs2.txt")
eigs <- sub("eig-", "", sub("-abs2.txt", "", eigfiles))
lapply(eigs, plotState)
