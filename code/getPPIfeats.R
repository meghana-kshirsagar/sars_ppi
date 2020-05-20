
library("rlist") 
library("protr")
library(parallel)
library(data.table)

print("Format: <ppi-file>  <fasta-file>  <outprefix>\n")
args <- commandArgs(trailingOnly=TRUE)
out_prefix = args[3]
fasta_file = args[2]
ppi_file = args[1]

ppis = read.csv(ppi_file,sep='\t',header=TRUE)

prots=readFASTA(fasta_file)

# generate features for all proteins
prots <- prots[(sapply(prots, protcheck))] 

prots.names = read.table("../data/all_proteinids.txt",header=F)
prots.names=prots.names$V1

prots.names = unlist(lapply(list.names(prots),function(x) { 
      if(grepl("\\|",x)) {
	strsplit(x,"\\|")[[1]][2] 
      }
      else {
	x
      }
	} ))

print(length(prots))

one.mer <- t(sapply(prots, extractAAC))

two.mer <- t(sapply(prots, extractDC))

tri.mer <- t(sapply(prots, extractTC))

ctriad <- t(sapply(prots, extractCTriad))

feats = data.frame(cbind(one.mer,two.mer,tri.mer,ctriad),row.names=1:nrow(one.mer))

dim(feats)


p1list = unlist(lapply(1:nrow(ppis), function(i) {
   idx = which(prots.names==ppis[i,1])
   if(length(idx)==0) {
      idx = 0
   }
   idx
 }))

p2list = unlist(lapply(1:nrow(ppis), function(i) {
   idx = which(prots.names==ppis[i,2])
   if(length(idx)==0) {
      idx = 0
   }
   idx
 }))

#### get only human protein features
ppis=read.csv("features/krogan_ppis_good.csv")
krogan_pos=ppis$V2
p1list = unlist(lapply(1:nrow(hprots), function(i) {
  if(!(hprots[i,1] %in% krogan_pos)) {
   idx = which(prots.names==hprots[i,1])
   if(length(idx)==0) { idx = 0 }
  }
  else { idx = 0 }
  idx
 }))

good.ppis=which(p1list>0)
p1list=p1list[good.ppis]
p1feats=feats[p1list,]

#################################

good.ppis = intersect(which(p1list>0) , which(p2list>0))
length(good.ppis)

p1list=p1list[good.ppis]
p2list=p2list[good.ppis]

p1feats=feats[p1list,]
p2feats=feats[p2list,]

feats.mat = cbind(p1feats,p2feats)

dim(feats.mat)
#feats.mat <- Reduce("rbind", res)

good.ppis = ppis[good.ppis,]

write.table(feats.mat,file=sprintf("%s_feats.csv",out_prefix),sep=",",row.names=F)
write.table(good.ppis,file=sprintf("%s.csv",out_prefix),sep=",")



