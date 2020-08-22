
cat.trans <- function(train_cat_x)
{
  
  cat_aov <- aov(sales_price %>% as.double ~ ., train_cat)
  x <- summary(cat_aov)[[1]]
  pval <- cbind(rownames(x),
                dplyr::select( x,`Pr(>F)`)) %>% data.frame
  
  pval2 <- filter(pval, pval[,2] < .05 ) %>% data.frame
  train_cat <- train_cat[,pval2[,1]]
  
  return(train_cat)
}