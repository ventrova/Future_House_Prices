#function file for preprocessing categorical data
#in ProjectR_final.R
#

bin.cat <- function(train_cat_x)
{

#dataframe for transformed predictors
trans_cat <- data.frame()
#number of categorical columns in original set
count_cat <- as.integer(count(data.frame(names(train_cat_x))))

for (i in 1:count_cat ) {
  
  #saves column levels in dataframe
  q <- data.frame(levels(train_cat_x[,names(train_cat_x[i])]))
  #number of levels
  countq <- as.integer(count(q))
  
  #for each level, create new column in transformed set
  for (t in 1 : countq) {
    a <- as.character(q[t,])
    trans_cat[1, paste(a,names(train_cat_x[i]))] <- 0
  }
  
}

#####################################################################
#fill all transformed rows to 0
trans_cat[1:nrow(train_cat_x),] <-0 
#
##count_trans <- as.integer(count(data.frame(names(trans_cat))))

# for each column, for each row, if value is in 
# names(transformed set), then replace 0 with 1 
# that row with the value as the column name
for (j in 1 : count_cat) {
  for (i in 1 : as.integer(count(train_cat_x))) {
    f <- train_cat_x[i,j]
    g <- names(train_cat_x[j])
    fg <- paste(as.character(f),g)
    
    if (fg %in% names(trans_cat)) {
      trans_cat[i,fg] = 1;
    }
  }
}



return(trans_cat)
}
