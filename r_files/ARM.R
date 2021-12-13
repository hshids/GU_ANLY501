library(networkD3)
library(arules)
library(jsonlite)
library(streamR)
library(rjson)
library(tokenizers)
library(tidyverse)
library(plyr)
library(dplyr)
library(ggplot2)
library(syuzhet)
library(stringr)
library(arulesViz)
library(igraph)
library(httpuv)
library(openssl)
library(tidytext)
library(igraph)
library(visNetwork)
library(networkD3)
library(magrittr)
setwd("/Users/hanjingshi/Desktop/GGT/arm")
TweetDF <- read.csv('tweet.csv',stringsAsFactors = FALSE)
Trans <- file(TransactionTweetsFile, open = "a")
for(i in 1:nrow(TweetDF)){
  Tokens<-tokenize_words(TweetDF$OriginalTweet[i],
                         stopwords = stopwords::stopwords("en"), 
                         lowercase = T,  
                         strip_punct = T, 
                         simplify = T)
  cat(unlist(Tokens), "\n", file=Trans, sep=",")
  cat(unlist(Tokens))
}
close(Trans)


