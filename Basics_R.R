        ########### workshop Machine Learning with R ########
        ###########    Dr Milan Joshi      #################
        ########        Basics Of R      ##### 
# Let R teach you b       
library(swirl)
install_from_swirl("R Programming")
setwd("")                               # Setting Working Directory
x <- 1                                  # Nothing Printed
x=1                                     # Nothing Printed
print(x)                                # Explicit Printing
x                                       # Auto Printing
msg <- "hello"
msg                                     # Auto Printing
x <-                                    # Incomplete expression
x <- 10:30                              # Create vector from 10 to 30
x
rm(x)
# Subsetting a vector
x>23
x[x>23]
which.max(x)
which.min(x)
min(x)
max(x)
x[x>min(x)+2]
x[x>23]<-NA
x
x <- 10:30
length(x)
x[x>20]<-rep(0,length(x)-20)
x
y<-1:6
y[y>3]<-c("A","B","C")
y
y<-rep(c(4,5,6),each=8)
unique(y)  # Duplicate values
##Subsetting a Vector
x <- c("a", "b", "c", "c", "d", "a")
x[1] ## Extract the first element
x[2] ## Extract the second element
x[1:4]
x[c(1, 3, 4)]
u <- x > "a"
u
x[u]
x[x > "a"]
#Subsetting a Matrix
x <- matrix(1:6, 2, 3)
x
x[1, 2]
x[2, 1]
x[1, ] ## Extract the first row
x[, 2] ## Extract the second column
#Subsetting Lists
x <- list(foo = 1:4, bar = 0.6)
x
x[[1]]
x[1]
x[["bar"]]
x$bar
x[[1]][[3]] ## Get the 3rd element of the 1st element
## 3rd element of the 1st element
x[[c(1, 3)]]
#Removing NA Values
x <- c(1, 2, NA, 4, NA, 5)
bad <- is.na(x)
sum(is.na(x))
print(bad)
x[!bad]
#For Multiplr Variables
x <- c(1, 2, NA, 4, NA, 5)
y <- c("a", "b", NA, "d", NA, "f")
good <- complete.cases(x, y)
good
x[good]
y[good]


################################################################################

###############################################################################
#Vector operations
x <- 1:4
y <- 6:9
z <- x + y # componantwise addition
z
a=log(10)
a
b=log2(c(1,2,3,4))
b
c=exp(2)
c
d=sqrt(88)
d
f=factorial(8)
f
r=choose(12,8)
rm(list=ls())
round(log(10), digits=3)
abs(18 / -12)
sin(3)

c <- (2 + sqrt(2))/(exp(2)+1)
c
x <- 1:10
y<-sample(length(x))
y
length(y)==length(x)
sum(x) # 1+2+3+4+---+10
sum(x[x>=2]) # 2+3+4+--+10
which(x>2)
any(x <=3)
which((x<=3))
mean=sum(x)/length(x)
mean
mean(x)
mean(x[x>5])
scale(x)
x > 2
x >= 2
x[x>2]
x - y
x * y # Componantwise multiplication
x / y
x %*% y # dot product Matrix Multiplication

x <- matrix(1:4, 2, 2)
y <- matrix(rep(10, 4), 2, 2)

## element-wise multiplication
x * y
## element-wise division
x / y
## true matrix multiplication
x %*% y
#Dates and Times
# Dates are stored internally as the
#number of days since 1970-01-01 while times 
#are stored internally as the number of seconds since 1970-01-01.

## Coerce a 'Date' object from character
x <- as.Date("1970-01-01")
x
23>90
0.5==0.5
0.5-0.3==0.4-0.2
x<-seq(1,5,by = 2)
x
x<-seq(1,5,by = 0.5)
x
x<-seq(0,1,length.out = 11)
x
x<-seq(5,1)
x
#Creating Vectors#
        
# The c() function can be used to create vectors of objects 
# by concatenating things together.
        
x <- c(0.5, 0.6)                         # numeric
x <- c(TRUE, FALSE)                      # logical
x <- c(T, F)                             # logical
        
# T and F are short-hand ways to specify TRUE and FALSE
x <- c("a", "b", "c")                    # character
x <- 9:29                                # integer
x <- c(1+0i, 2+4i)                       # complex
        
#You can also use the vector() function to initialize vectors.
        
x <- vector("numeric", length = 10)
x
### Mixing Objects ###
y <- c(1.7, "a")                         # character
y <- c(TRUE, 2)                          # numeric
y <- c("a", TRUE)                        # character
###Explicit Coercion###
        
#Objects can be explicitly coerced from one class to another 
#using the as.* functions, if available.
        
x <- 0:6
class(x)
as.numeric(x)
as.logical(x)
as.character(x)
#Sometimes, R can't figure out how to coerce an object 
#and this can result in NAs being produced.
x <- c("a", "b", "c")
as.numeric(x)
as.logical(x)
as.complex(x)
##Matrices
#Matrices are vectors with a dimension attribute. 
#The dimension attribute is itself an integer vector
#of length 2 (number of rows, number of columns)
m<-matrix(nrow = 2, ncol = 3)
m
dim(m)
attributes(m)
#Matrices are constructed column-wise, 
#so entries can be thought of starting in the "upper left" corner
#and running down the columns.
m <- matrix(1:6, nrow = 2, ncol = 3)
m
#Matrices can also be created directly from vectors by 
#adding a dimension attribute.
m <- 1:10
m
dim(m) <- c(2, 5)
m
#Matrices can be created by column-binding or row-binding with the 
#cbind() and rbind() functions.
x <- 1:3
x
y <- 10:12
y
cbind(x, y)   # First column of x and 2nd of y
rbind(x, y)   # First row of x and 2nd of y
#######Lists
#Lists are a special type of vector that can contain elements of 
#different classes. Lists are a very important data type in R
        
x <- list(1, "a", TRUE, 1 + 4i)
x
#We can also create an empty list of a prespecified length 
#with the vector() function
x <- vector("list", length = 5)
x
#Factors
#Factors are used to represent categorical data
#and can be unordered or ordered.
x <- factor(c("yes", "yes", "no", "yes", "no"))
x
table(x)
#See the underlying representation of factor
unclass(x)
# rep in r      
x<-rep(4,5)              # Repeats four five time
x<-rep(c(0,1),20)        # Repeats 0,1 twenty times
x
x<-rep(c(0,1),c(5,5))    # repeats 0 five times and 1 five times
x
class(x)
y<-as.factor(x)          # converting x to factor
y
        
#Missing Values
#Missing values are denoted by NA or NaN for 
#undefined mathematical operations.
#. is.na() is used to test objects if they are NA
#. is.nan() is used to test for NaN
#. NA values have a class also, so there are integer NA, character NA, etc.
#. A NaN value is also NA but the converse is not true
## Create a vector with NAs in it
x <- c(1, 2, NA, 10, 3)
## Return a logical vector indicating which elements are NA
is.na(x)
#Return a logical vector indicating which elements are NaN
is.nan(x)
## Now create a vector with both NA and NaN values
x <- c(1, 2, NaN, NA, 4)
is.na(x)
is.nan(x)
# Make up a randomly ordered vector
v <- sample(101:110)
v
# Sort the vector
sort(v)
# Reverse sort
sort(v, decreasing=TRUE)
##############################
LETTERS
letters
letters[1:6]
x<-sample(letters[1:9],10)
x
x<-sample(letters[1:9],replace=T,10)
x
x<-sample(LETTERS,10)
x
##################################


#############Data Frames############
# Data frames are used to store tabular data in R. 
# They are an important type of object in R and are
# used in a variety of statistical modeling applications.
# Hadley Wickham's package dplyr has an optimized set of functions 
# designed to work efficiently with data frames.
# data frames have a special attribute called row.names 
# which indicate information about each row of the data frame.
# Data frames are usually created by reading in a dataset using 
# the read.table() or read.csv().However, data frames can also be 
# created explicitly with the data.frame() function or they can be
# coerced from other types of objects like lists.
# Data frames can be converted to a matrix by calling data.matrix().
# While it might seem that the as.matrix() function should be used
# to coerce a data frame to a matrix, almost always, what you
# want is the result of data.matrix().
x <- data.frame(foo = 1:4, bar = c(T, T, F, F)) # creating data frame
x
dim(x)
str(x)
class(x)
names(x)                        # Name of each Variable
row.names(x)                    # Name of Rows
row.names(x)<-c(LETTERS[1:4])   # Convert Row names to A,B,C,D
as.matrix(x)
x
x[,1]       # x[row,column] All rows first column
x[1,]       # First row all columns
x[-1,]      # Exclude First row
x[-c(1,2),] # Exclude First two rows
x[,-2]      # exclude 2 column
        
#Names
x <- 1:3
names(x)  # No names 
names(x) <- c("New York", "Seattle", "Los Angeles")
x
names(x)
#Lists can also have names, which is often very useful.
x <- list("Los Angeles" = 1, Boston = 2, London = 3)
x
names(x)
#Matrices can have both column and row names.
m <- matrix(1:4, nrow = 2, ncol = 2)
dimnames(m) <- list(c("a", "b"), c("c", "d"))
m
#Column names and row names can be set separately 
#using the colnames() and rownames() functions.
colnames(m) <- c("h", "f")
rownames(m) <- c("x", "z")
m
        
##  Object     Set column names     Set row names
## data frame    names()             row.names()
## matrix       colnames()            rownames()
        
##########Reading and Writing Data#########
#read.table, read.csv, for reading tabular data
#. readLines, for reading lines of a text file
# Many packages in R to read All kind of datsets
# write.table, for writing tabular data to text files (i.e. CSV) or connections
? read.table()
data <- read.table("foo.txt")  #Reading txt file named foo
initial <- read.table("datatable.txt", nrows = 100)
#############################################
a <- data.frame(x = rnorm(100), y = runif(100))
b <- c(3, 4.4, 1 / 3)
## Save 'a' and 'b' to a file
save(a, b, file = "mydata.rda")
## Load 'a' and 'b' into your workspace
load("mydata.rda")
##############################################
# load the library
library(RCurl)
# specify the URL for the Iris data CSV
urlfile <-'https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data'
# download the file
downloaded <- getURL(urlfile, ssl.verifypeer=FALSE)
# treat the text data as a steam so we can read from it
connection <- textConnection(downloaded)
# parse the downloaded data as CSV
balance<- read.csv(connection, header=FALSE)
######################################################
library(data.table)
mydat <- fread('http://www.stats.ox.ac.uk/pub/datasets/csb/ch11b.dat')
head(mydat)
######################################################

#Control Structures
#if-else
if(<condition>) {
    ## do something
        }
## Continue with rest of code
#######################################
        
if(<condition>) {
  ## do something
  }
else {
## do something else
  }
##########################################
        
if(<condition1>) {
## do something
} else if(<condition2>) {
 ## do something different
} else {
## do something different
}
########################################
        
## Generate a uniform random number
x <- runif(1, 0, 10)
if(x > 3) {
  y <- 10
} else {
y <- 0
 }
 y
#####################################
y <- if(x > 3) {
          10
} else {
          0
}
#####################################
        # For Loops
        for(i in 1:10) {
          print(i^2)
        }
        #####################################
        x <- c("a", "b", "c", "d")
        for(i in 1:4) {
          ## Print out each element of 'x'
          print(x)
        }
        ##################################
        for(i in seq_along(x)) {
          print(x[i])
        }
        ####################################
        for(letter in x) {
          print(letter)
        }
        ##################################
        for(i in 1:4) print(x[i])
        ##################################
        #Nested For Loops
        x <- matrix(1:6, 2, 3)
        for(i in seq_len(nrow(x))) {
          for(j in seq_len(ncol(x))) {
            print(x[i, j])
          }
        }
        x[1,3]
        #######################################
        
        # if and for on iris data
        x<-iris$Sepal.Length
        head(x,10)
        tail(x)
        mean(x)
        for(i in 1:length(x)){
          if(x[i]>mean(x)){
            x[i]<-0
          }
        }
        x
#while Loops
        count <- 0
        while(count < 10) {
          print(count)
          count <- count + 1
        }
        #################################
        # Functions in R
        
        f <- function() {
          ## This is an empty function
        }
        ## Functions have their own class
        class(f)
        #############################
        f <- function(a,b) {
          a+b}
        f(2,4)
        ##################
        f <- function(a,b) {
          a^2+b^2}
        f(2,3)
        #######################
        stat<-function(x){ a=mean(x)
        b=sd(x)
        return(c(a,b))}
        stat(1:4)
        
        
        y <- 10
        f <- function(x) {
          y <- 2
          y^2 + g(x)
        }
        g <- function(x) {
          x*y
        }
        f(3)
        ###################################################
        f.good <- function(x, y) {
          z1 <- 2*x + y
          z2 <- x + 2*y
          z3 <- 2*x + 2*y
          z4 <- x/y
          return(c(z1, z2, z3, z4))
        }
        ###################################################
        f<- function(x) {
          for(i in 1:x) {
            y <- i*2
            print(y)
          }
          return(y*2)
        }	
        f(3)
        ##########################################################
        
data(iris)
class(iris)
mean(iris$Sepal.Length)
sd(iris$Sepal.Length)
        var(iris$Sepal.Length)
        min(iris$Sepal.Length)
        max(iris$Sepal.Length)
        median(iris$Sepal.Length)
        range(iris$Sepal.Length)
        quantile(iris$Sepal.Length)
        summary(iris)
        t<-table(iris$Species)
        t
        prp<-prop.table(t)
        prp
        round(prp,2)
        bin<-cut(iris$Sepal.Length, 10, ordered = TRUE)
        table(bin)
        barplot(t,col="blue")  # barplot of %
        ###############################################################################################
        #################################
        #https://www.r-bloggers.com/hands-on-dplyr-tutorial-
          #for-faster-data-manipulation-in-r/
       # dplyr(filter select arrange mutate summrise)
        
        library(dplyr)
        mydata<-read.csv("sampledata.csv")
        one<-sample_n(mydata,3)
        two<-sample_frac(mydata,0.1)
        x1 = distinct(mydata) # Remove duplicate Rows
        mydata2 = select(mydata, Index, State:Y2008)
        #Selecting or Dropping Variables starts with 'Y'
        mydata3 = select(mydata, starts_with("Y"))
        mydata6 = rename(mydata, Index1=Index)
        mydata7 = filter(mydata, Index == "A")
        mydata7 = filter(mydata6, Index %in% c("A", "C"))
        mydata8 = filter(mydata6, Index %in% c("A", "C") & Y2002 >= 1300000 )
        mydata9 = filter(mydata6, Index %in% c("A", "C") | Y2002 >= 1300000)
        mydata10 = filter(mydata6, !Index %in% c("A", "C"))
        dt = mydata %>% select(Index, State) %>% sample_n(10) # pipe operator
        
    ###############################################################3
        
        ############base graphics
        
        ######################################################
        #   Loop Functions
        #  lapply(): Loop over a list and evaluate a function on each element
        #. sapply(): Same as lapply but try to simplify the result
        #. apply(): Apply a function over the margins of an array
        #. tapply(): Apply a function over subsets of a vector
        #. mapply(): Multivariate version of lapply
        x <- matrix(rnorm(200), 20, 10)
        apply(x, 2, mean)  ## Take the mean of each column
        apply(x, 1, sum)  ## Take the sum of each row
        sapply(airquality, function(x) sum(is.na(x)))
        s <- split(airquality, airquality$Month)
        sapply(s, function(x){ colMeans(x[,c("Ozone", "Solar.R", "Wind")])})
        #############################################################################
        # Loading Data
        
        Auto=read.table("Auto.data")
        fix(Auto)
        Auto=read.table("Auto.data",header=T,na.strings="?")
        fix(Auto)
        Auto=read.csv("Auto.csv",header=T,na.strings="?")
        fix(Auto)
        dim(Auto)
        Auto[1:4,]
        Auto=na.omit(Auto)
        dim(Auto)
        names(Auto)
        
        # Additional Graphical and Numerical Summaries
        
        plot(cylinders, mpg)
        plot(Auto$cylinders, Auto$mpg)
        attach(Auto)
        plot(cylinders, mpg)
        cylinders=as.factor(cylinders)
        plot(cylinders, mpg)
        plot(cylinders, mpg, col="red")
        plot(cylinders, mpg, col="red", varwidth=T)
        plot(cylinders, mpg, col="red", varwidth=T,horizontal=T)
        plot(cylinders, mpg, col="red", varwidth=T, xlab="cylinders", ylab="MPG")
        hist(mpg)
        hist(mpg,col=2)
        hist(mpg,col=2,breaks=15)
        pairs(Auto)
        pairs(~ mpg + displacement + horsepower + weight + acceleration, Auto)
        plot(horsepower,mpg)
        identify(horsepower,mpg,name)
        summary(Auto)
        summary(mpg)
        
        
        
        # Chapter 3 Lab: Linear Regression
        
        library(MASS)
        library(ISLR)
        
        # Simple Linear Regression
        
        fix(Boston)
        names(Boston)
        lm.fit=lm(medv~lstat)
        lm.fit=lm(medv~lstat,data=Boston)
        attach(Boston)
        lm.fit=lm(medv~lstat)
        lm.fit
        summary(lm.fit)
        names(lm.fit)
        coef(lm.fit)
        confint(lm.fit)
        predict(lm.fit,data.frame(lstat=(c(5,10,15))), interval="confidence")
        predict(lm.fit,data.frame(lstat=(c(5,10,15))), interval="prediction")
        plot(lstat,medv)
        abline(lm.fit)
        abline(lm.fit,lwd=3)
        abline(lm.fit,lwd=3,col="red")
        plot(lstat,medv,col="red")
        plot(lstat,medv,pch=20)
        plot(lstat,medv,pch="+")
        plot(1:20,1:20,pch=1:20)
        par(mfrow=c(2,2))
        plot(lm.fit)
        plot(predict(lm.fit), residuals(lm.fit))
        plot(predict(lm.fit), rstudent(lm.fit))
        plot(hatvalues(lm.fit))
        which.max(hatvalues(lm.fit))
        
        # Multiple Linear Regression
        
        lm.fit=lm(medv~lstat+age,data=Boston)
        summary(lm.fit)
        lm.fit=lm(medv~.,data=Boston)
        summary(lm.fit)
        library(car)
        vif(lm.fit)
        lm.fit1=lm(medv~.-age,data=Boston)
        summary(lm.fit1)
        lm.fit1=update(lm.fit, ~.-age)
        
        # Interaction Terms
        
        summary(lm(medv~lstat*age,data=Boston))
        
        # Non-linear Transformations of the Predictors
        
        lm.fit2=lm(medv~lstat+I(lstat^2))
        summary(lm.fit2)
        lm.fit=lm(medv~lstat)
        anova(lm.fit,lm.fit2)
        par(mfrow=c(2,2))
        plot(lm.fit2)
        lm.fit5=lm(medv~poly(lstat,5))
        summary(lm.fit5)
        summary(lm(medv~log(rm),data=Boston))
        
        # Qualitative Predictors
        
        fix(Carseats)
        names(Carseats)
        lm.fit=lm(Sales~.+Income:Advertising+Price:Age,data=Carseats)
        summary(lm.fit)
        attach(Carseats)
        contrasts(ShelveLoc)
        
        # Writing Functions
        
        LoadLibraries
        LoadLibraries()
        LoadLibraries=function(){
          library(ISLR)
          library(MASS)
          print("The libraries have been loaded.")
        }
        LoadLibraries
        LoadLibraries()
        
        
        # Chapter 4 Lab: Logistic Regression, LDA, QDA, and KNN
        
        # The Stock Market Data
        
        library(ISLR)
        names(Smarket)
        dim(Smarket)
        summary(Smarket)
        pairs(Smarket)
        cor(Smarket)
        cor(Smarket[,-9])
        attach(Smarket)
        plot(Volume)
        
        # Logistic Regression
        
        glm.fit=glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume,data=Smarket,family=binomial)
        summary(glm.fit)
        coef(glm.fit)
        summary(glm.fit)$coef
        summary(glm.fit)$coef[,4]
        glm.probs=predict(glm.fit,type="response")
        glm.probs[1:10]
        contrasts(Direction)
        glm.pred=rep("Down",1250)
        glm.pred[glm.probs>.5]="Up"
        table(glm.pred,Direction)
        (507+145)/1250
        mean(glm.pred==Direction)
        train=(Year<2005)
        Smarket.2005=Smarket[!train,]
        dim(Smarket.2005)
        Direction.2005=Direction[!train]
        glm.fit=glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume,data=Smarket,family=binomial,subset=train)
        glm.probs=predict(glm.fit,Smarket.2005,type="response")
        glm.pred=rep("Down",252)
        glm.pred[glm.probs>.5]="Up"
        table(glm.pred,Direction.2005)
        mean(glm.pred==Direction.2005)
        mean(glm.pred!=Direction.2005)
        glm.fit=glm(Direction~Lag1+Lag2,data=Smarket,family=binomial,subset=train)
        glm.probs=predict(glm.fit,Smarket.2005,type="response")
        glm.pred=rep("Down",252)
        glm.pred[glm.probs>.5]="Up"
        table(glm.pred,Direction.2005)
        mean(glm.pred==Direction.2005)
        106/(106+76)
        predict(glm.fit,newdata=data.frame(Lag1=c(1.2,1.5),Lag2=c(1.1,-0.8)),type="response")
        
        # Linear Discriminant Analysis
        
        library(MASS)
        lda.fit=lda(Direction~Lag1+Lag2,data=Smarket,subset=train)
        lda.fit
        plot(lda.fit)
        lda.pred=predict(lda.fit, Smarket.2005)
        names(lda.pred)
        lda.class=lda.pred$class
        table(lda.class,Direction.2005)
        mean(lda.class==Direction.2005)
        sum(lda.pred$posterior[,1]>=.5)
        sum(lda.pred$posterior[,1]<.5)
        lda.pred$posterior[1:20,1]
        lda.class[1:20]
        sum(lda.pred$posterior[,1]>.9)
        
        # Quadratic Discriminant Analysis
        
        qda.fit=qda(Direction~Lag1+Lag2,data=Smarket,subset=train)
        qda.fit
        qda.class=predict(qda.fit,Smarket.2005)$class
        table(qda.class,Direction.2005)
        mean(qda.class==Direction.2005)
        
        # K-Nearest Neighbors
        
        library(class)
        train.X=cbind(Lag1,Lag2)[train,]
        test.X=cbind(Lag1,Lag2)[!train,]
        train.Direction=Direction[train]
        set.seed(1)
        knn.pred=knn(train.X,test.X,train.Direction,k=1)
        table(knn.pred,Direction.2005)
        (83+43)/252
        knn.pred=knn(train.X,test.X,train.Direction,k=3)
        table(knn.pred,Direction.2005)
        mean(knn.pred==Direction.2005)
        
        # An Application to Caravan Insurance Data
        
        dim(Caravan)
        attach(Caravan)
        summary(Purchase)
        348/5822
        standardized.X=scale(Caravan[,-86])
        var(Caravan[,1])
        var(Caravan[,2])
        var(standardized.X[,1])
        var(standardized.X[,2])
        test=1:1000
        train.X=standardized.X[-test,]
        test.X=standardized.X[test,]
        train.Y=Purchase[-test]
        test.Y=Purchase[test]
        set.seed(1)
        knn.pred=knn(train.X,test.X,train.Y,k=1)
        mean(test.Y!=knn.pred)
        mean(test.Y!="No")
        table(knn.pred,test.Y)
        9/(68+9)
        knn.pred=knn(train.X,test.X,train.Y,k=3)
        table(knn.pred,test.Y)
        5/26
        knn.pred=knn(train.X,test.X,train.Y,k=5)
        table(knn.pred,test.Y)
        4/15
        glm.fit=glm(Purchase~.,data=Caravan,family=binomial,subset=-test)
        glm.probs=predict(glm.fit,Caravan[test,],type="response")
        glm.pred=rep("No",1000)
        glm.pred[glm.probs>.5]="Yes"
        table(glm.pred,test.Y)
        glm.pred=rep("No",1000)
        glm.pred[glm.probs>.25]="Yes"
        table(glm.pred,test.Y)
        11/(22+11)
        
        
        
        # Chaper 5 Lab: Cross-Validation and the Bootstrap
        
        # The Validation Set Approach
        
        library(ISLR)
        set.seed(1)
        train=sample(392,196)
        lm.fit=lm(mpg~horsepower,data=Auto,subset=train)
        attach(Auto)
        mean((mpg-predict(lm.fit,Auto))[-train]^2)
        lm.fit2=lm(mpg~poly(horsepower,2),data=Auto,subset=train)
        mean((mpg-predict(lm.fit2,Auto))[-train]^2)
        lm.fit3=lm(mpg~poly(horsepower,3),data=Auto,subset=train)
        mean((mpg-predict(lm.fit3,Auto))[-train]^2)
        set.seed(2)
        train=sample(392,196)
        lm.fit=lm(mpg~horsepower,subset=train)
        mean((mpg-predict(lm.fit,Auto))[-train]^2)
        lm.fit2=lm(mpg~poly(horsepower,2),data=Auto,subset=train)
        mean((mpg-predict(lm.fit2,Auto))[-train]^2)
        lm.fit3=lm(mpg~poly(horsepower,3),data=Auto,subset=train)
        mean((mpg-predict(lm.fit3,Auto))[-train]^2)
        
        # Leave-One-Out Cross-Validation
        
        glm.fit=glm(mpg~horsepower,data=Auto)
        coef(glm.fit)
        lm.fit=lm(mpg~horsepower,data=Auto)
        coef(lm.fit)
        library(boot)
        glm.fit=glm(mpg~horsepower,data=Auto)
        cv.err=cv.glm(Auto,glm.fit)
        cv.err$delta
        cv.error=rep(0,5)
        for (i in 1:5){
          glm.fit=glm(mpg~poly(horsepower,i),data=Auto)
          cv.error[i]=cv.glm(Auto,glm.fit)$delta[1]
        }
        cv.error
        
        # k-Fold Cross-Validation
        
        set.seed(17)
        cv.error.10=rep(0,10)
        for (i in 1:10){
          glm.fit=glm(mpg~poly(horsepower,i),data=Auto)
          cv.error.10[i]=cv.glm(Auto,glm.fit,K=10)$delta[1]
        }
        cv.error.10
        
        # The Bootstrap
        
        alpha.fn=function(data,index){
          X=data$X[index]
          Y=data$Y[index]
          return((var(Y)-cov(X,Y))/(var(X)+var(Y)-2*cov(X,Y)))
        }
        alpha.fn(Portfolio,1:100)
        set.seed(1)
        alpha.fn(Portfolio,sample(100,100,replace=T))
        boot(Portfolio,alpha.fn,R=1000)
        
        # Estimating the Accuracy of a Linear Regression Model
        
        boot.fn=function(data,index)
          return(coef(lm(mpg~horsepower,data=data,subset=index)))
        boot.fn(Auto,1:392)
        set.seed(1)
        boot.fn(Auto,sample(392,392,replace=T))
        boot.fn(Auto,sample(392,392,replace=T))
        boot(Auto,boot.fn,1000)
        summary(lm(mpg~horsepower,data=Auto))$coef
        boot.fn=function(data,index)
          coefficients(lm(mpg~horsepower+I(horsepower^2),data=data,subset=index))
        set.seed(1)
        boot(Auto,boot.fn,1000)
        summary(lm(mpg~horsepower+I(horsepower^2),data=Auto))$coef
        
        
        
        # Chapter 6 Lab 1: Subset Selection Methods
        
        # Best Subset Selection
        
        library(ISLR)
        fix(Hitters)
        names(Hitters)
        dim(Hitters)
        sum(is.na(Hitters$Salary))
        Hitters=na.omit(Hitters)
        dim(Hitters)
        sum(is.na(Hitters))
        library(leaps)
        regfit.full=regsubsets(Salary~.,Hitters)
        summary(regfit.full)
        regfit.full=regsubsets(Salary~.,data=Hitters,nvmax=19)
        reg.summary=summary(regfit.full)
        names(reg.summary)
        reg.summary$rsq
        par(mfrow=c(2,2))
        plot(reg.summary$rss,xlab="Number of Variables",ylab="RSS",type="l")
        plot(reg.summary$adjr2,xlab="Number of Variables",ylab="Adjusted RSq",type="l")
        which.max(reg.summary$adjr2)
        points(11,reg.summary$adjr2[11], col="red",cex=2,pch=20)
        plot(reg.summary$cp,xlab="Number of Variables",ylab="Cp",type='l')
        which.min(reg.summary$cp)
        points(10,reg.summary$cp[10],col="red",cex=2,pch=20)
        which.min(reg.summary$bic)
        plot(reg.summary$bic,xlab="Number of Variables",ylab="BIC",type='l')
        points(6,reg.summary$bic[6],col="red",cex=2,pch=20)
        plot(regfit.full,scale="r2")
        plot(regfit.full,scale="adjr2")
        plot(regfit.full,scale="Cp")
        plot(regfit.full,scale="bic")
        coef(regfit.full,6)
        
        # Forward and Backward Stepwise Selection
        
        regfit.fwd=regsubsets(Salary~.,data=Hitters,nvmax=19,method="forward")
        summary(regfit.fwd)
        regfit.bwd=regsubsets(Salary~.,data=Hitters,nvmax=19,method="backward")
        summary(regfit.bwd)
        coef(regfit.full,7)
        coef(regfit.fwd,7)
        coef(regfit.bwd,7)
        
        # Choosing Among Models
        
        set.seed(1)
        train=sample(c(TRUE,FALSE), nrow(Hitters),rep=TRUE)
        test=(!train)
        regfit.best=regsubsets(Salary~.,data=Hitters[train,],nvmax=19)
        test.mat=model.matrix(Salary~.,data=Hitters[test,])
        val.errors=rep(NA,19)
        for(i in 1:19){
          coefi=coef(regfit.best,id=i)
          pred=test.mat[,names(coefi)]%*%coefi
          val.errors[i]=mean((Hitters$Salary[test]-pred)^2)
        }
        val.errors
        which.min(val.errors)
        coef(regfit.best,10)
        predict.regsubsets=function(object,newdata,id,...){
          form=as.formula(object$call[[2]])
          mat=model.matrix(form,newdata)
          coefi=coef(object,id=id)
          xvars=names(coefi)
          mat[,xvars]%*%coefi
        }
        regfit.best=regsubsets(Salary~.,data=Hitters,nvmax=19)
        coef(regfit.best,10)
        k=10
        set.seed(1)
        folds=sample(1:k,nrow(Hitters),replace=TRUE)
        cv.errors=matrix(NA,k,19, dimnames=list(NULL, paste(1:19)))
        for(j in 1:k){
          best.fit=regsubsets(Salary~.,data=Hitters[folds!=j,],nvmax=19)
          for(i in 1:19){
            pred=predict(best.fit,Hitters[folds==j,],id=i)
            cv.errors[j,i]=mean( (Hitters$Salary[folds==j]-pred)^2)
          }
        }
        mean.cv.errors=apply(cv.errors,2,mean)
        mean.cv.errors
        par(mfrow=c(1,1))
        plot(mean.cv.errors,type='b')
        reg.best=regsubsets(Salary~.,data=Hitters, nvmax=19)
        coef(reg.best,11)
        
        
        # Chapter 6 Lab 2: Ridge Regression and the Lasso
        
        x=model.matrix(Salary~.,Hitters)[,-1]
        y=Hitters$Salary
        
        # Ridge Regression
        
        library(glmnet)
        grid=10^seq(10,-2,length=100)
        ridge.mod=glmnet(x,y,alpha=0,lambda=grid)
        dim(coef(ridge.mod))
        ridge.mod$lambda[50]
        coef(ridge.mod)[,50]
        sqrt(sum(coef(ridge.mod)[-1,50]^2))
        ridge.mod$lambda[60]
        coef(ridge.mod)[,60]
        sqrt(sum(coef(ridge.mod)[-1,60]^2))
        predict(ridge.mod,s=50,type="coefficients")[1:20,]
        set.seed(1)
        train=sample(1:nrow(x), nrow(x)/2)
        test=(-train)
        y.test=y[test]
        ridge.mod=glmnet(x[train,],y[train],alpha=0,lambda=grid, thresh=1e-12)
        ridge.pred=predict(ridge.mod,s=4,newx=x[test,])
        mean((ridge.pred-y.test)^2)
        mean((mean(y[train])-y.test)^2)
        ridge.pred=predict(ridge.mod,s=1e10,newx=x[test,])
        mean((ridge.pred-y.test)^2)
        ridge.pred=predict(ridge.mod,s=0,newx=x[test,],exact=T)
        mean((ridge.pred-y.test)^2)
        lm(y~x, subset=train)
        predict(ridge.mod,s=0,exact=T,type="coefficients")[1:20,]
        set.seed(1)
        cv.out=cv.glmnet(x[train,],y[train],alpha=0)
        plot(cv.out)
        bestlam=cv.out$lambda.min
        bestlam
        ridge.pred=predict(ridge.mod,s=bestlam,newx=x[test,])
        mean((ridge.pred-y.test)^2)
        out=glmnet(x,y,alpha=0)
        predict(out,type="coefficients",s=bestlam)[1:20,]
        
        # The Lasso
        
        lasso.mod=glmnet(x[train,],y[train],alpha=1,lambda=grid)
        plot(lasso.mod)
        set.seed(1)
        cv.out=cv.glmnet(x[train,],y[train],alpha=1)
        plot(cv.out)
        bestlam=cv.out$lambda.min
        lasso.pred=predict(lasso.mod,s=bestlam,newx=x[test,])
        mean((lasso.pred-y.test)^2)
        out=glmnet(x,y,alpha=1,lambda=grid)
        lasso.coef=predict(out,type="coefficients",s=bestlam)[1:20,]
        lasso.coef
        lasso.coef[lasso.coef!=0]
        
        
        # Chapter 6 Lab 3: PCR and PLS Regression
        
        # Principal Components Regression
        
        library(pls)
        set.seed(2)
        pcr.fit=pcr(Salary~., data=Hitters,scale=TRUE,validation="CV")
        summary(pcr.fit)
        validationplot(pcr.fit,val.type="MSEP")
        set.seed(1)
        pcr.fit=pcr(Salary~., data=Hitters,subset=train,scale=TRUE, validation="CV")
        validationplot(pcr.fit,val.type="MSEP")
        pcr.pred=predict(pcr.fit,x[test,],ncomp=7)
        mean((pcr.pred-y.test)^2)
        pcr.fit=pcr(y~x,scale=TRUE,ncomp=7)
        summary(pcr.fit)
        
        # Partial Least Squares
        
        set.seed(1)
        pls.fit=plsr(Salary~., data=Hitters,subset=train,scale=TRUE, validation="CV")
        summary(pls.fit)
        validationplot(pls.fit,val.type="MSEP")
        pls.pred=predict(pls.fit,x[test,],ncomp=2)
        mean((pls.pred-y.test)^2)
        pls.fit=plsr(Salary~., data=Hitters,scale=TRUE,ncomp=2)
        summary(pls.fit)
        
        
        
        # Chapter 7 Lab: Non-linear Modeling
        
        library(ISLR)
        attach(Wage)
        
        # Polynomial Regression and Step Functions
        
        fit=lm(wage~poly(age,4),data=Wage)
        coef(summary(fit))
        fit2=lm(wage~poly(age,4,raw=T),data=Wage)
        coef(summary(fit2))
        fit2a=lm(wage~age+I(age^2)+I(age^3)+I(age^4),data=Wage)
        coef(fit2a)
        fit2b=lm(wage~cbind(age,age^2,age^3,age^4),data=Wage)
        agelims=range(age)
        age.grid=seq(from=agelims[1],to=agelims[2])
        preds=predict(fit,newdata=list(age=age.grid),se=TRUE)
        se.bands=cbind(preds$fit+2*preds$se.fit,preds$fit-2*preds$se.fit)
        par(mfrow=c(1,2),mar=c(4.5,4.5,1,1),oma=c(0,0,4,0))
        plot(age,wage,xlim=agelims,cex=.5,col="darkgrey")
        title("Degree-4 Polynomial",outer=T)
        lines(age.grid,preds$fit,lwd=2,col="blue")
        matlines(age.grid,se.bands,lwd=1,col="blue",lty=3)
        preds2=predict(fit2,newdata=list(age=age.grid),se=TRUE)
        max(abs(preds$fit-preds2$fit))
        fit.1=lm(wage~age,data=Wage)
        fit.2=lm(wage~poly(age,2),data=Wage)
        fit.3=lm(wage~poly(age,3),data=Wage)
        fit.4=lm(wage~poly(age,4),data=Wage)
        fit.5=lm(wage~poly(age,5),data=Wage)
        anova(fit.1,fit.2,fit.3,fit.4,fit.5)
        coef(summary(fit.5))
        (-11.983)^2
        fit.1=lm(wage~education+age,data=Wage)
        fit.2=lm(wage~education+poly(age,2),data=Wage)
        fit.3=lm(wage~education+poly(age,3),data=Wage)
        anova(fit.1,fit.2,fit.3)
        fit=glm(I(wage>250)~poly(age,4),data=Wage,family=binomial)
        preds=predict(fit,newdata=list(age=age.grid),se=T)
        pfit=exp(preds$fit)/(1+exp(preds$fit))
        se.bands.logit = cbind(preds$fit+2*preds$se.fit, preds$fit-2*preds$se.fit)
        se.bands = exp(se.bands.logit)/(1+exp(se.bands.logit))
        preds=predict(fit,newdata=list(age=age.grid),type="response",se=T)
        plot(age,I(wage>250),xlim=agelims,type="n",ylim=c(0,.2))
        points(jitter(age), I((wage>250)/5),cex=.5,pch="|",col="darkgrey")
        lines(age.grid,pfit,lwd=2, col="blue")
        matlines(age.grid,se.bands,lwd=1,col="blue",lty=3)
        table(cut(age,4))
        fit=lm(wage~cut(age,4),data=Wage)
        coef(summary(fit))
        
        # Splines
        
        library(splines)
        fit=lm(wage~bs(age,knots=c(25,40,60)),data=Wage)
        pred=predict(fit,newdata=list(age=age.grid),se=T)
        plot(age,wage,col="gray")
        lines(age.grid,pred$fit,lwd=2)
        lines(age.grid,pred$fit+2*pred$se,lty="dashed")
        lines(age.grid,pred$fit-2*pred$se,lty="dashed")
        dim(bs(age,knots=c(25,40,60)))
        dim(bs(age,df=6))
        attr(bs(age,df=6),"knots")
        fit2=lm(wage~ns(age,df=4),data=Wage)
        pred2=predict(fit2,newdata=list(age=age.grid),se=T)
        lines(age.grid, pred2$fit,col="red",lwd=2)
        plot(age,wage,xlim=agelims,cex=.5,col="darkgrey")
        title("Smoothing Spline")
        fit=smooth.spline(age,wage,df=16)
        fit2=smooth.spline(age,wage,cv=TRUE)
        fit2$df
        lines(fit,col="red",lwd=2)
        lines(fit2,col="blue",lwd=2)
        legend("topright",legend=c("16 DF","6.8 DF"),col=c("red","blue"),lty=1,lwd=2,cex=.8)
        plot(age,wage,xlim=agelims,cex=.5,col="darkgrey")
        title("Local Regression")
        fit=loess(wage~age,span=.2,data=Wage)
        fit2=loess(wage~age,span=.5,data=Wage)
        lines(age.grid,predict(fit,data.frame(age=age.grid)),col="red",lwd=2)
        lines(age.grid,predict(fit2,data.frame(age=age.grid)),col="blue",lwd=2)
        legend("topright",legend=c("Span=0.2","Span=0.5"),col=c("red","blue"),lty=1,lwd=2,cex=.8)
        
        # GAMs
        
        gam1=lm(wage~ns(year,4)+ns(age,5)+education,data=Wage)
        library(gam)
        gam.m3=gam(wage~s(year,4)+s(age,5)+education,data=Wage)
        par(mfrow=c(1,3))
        plot(gam.m3, se=TRUE,col="blue")
        plot.gam(gam1, se=TRUE, col="red")
        gam.m1=gam(wage~s(age,5)+education,data=Wage)
        gam.m2=gam(wage~year+s(age,5)+education,data=Wage)
        anova(gam.m1,gam.m2,gam.m3,test="F")
        summary(gam.m3)
        preds=predict(gam.m2,newdata=Wage)
        gam.lo=gam(wage~s(year,df=4)+lo(age,span=0.7)+education,data=Wage)
        plot.gam(gam.lo, se=TRUE, col="green")
        gam.lo.i=gam(wage~lo(year,age,span=0.5)+education,data=Wage)
        library(akima)
        plot(gam.lo.i)
        gam.lr=gam(I(wage>250)~year+s(age,df=5)+education,family=binomial,data=Wage)
        par(mfrow=c(1,3))
        plot(gam.lr,se=T,col="green")
        table(education,I(wage>250))
        gam.lr.s=gam(I(wage>250)~year+s(age,df=5)+education,family=binomial,data=Wage,subset=(education!="1. < HS Grad"))
        plot(gam.lr.s,se=T,col="green")
        
        
        
        # Chapter 8 Lab: Decision Trees
        
        # Fitting Classification Trees
        
        library(tree)
        library(ISLR)
        attach(Carseats)
        High=ifelse(Sales<=8,"No","Yes")
        Carseats=data.frame(Carseats,High)
        tree.carseats=tree(High~.-Sales,Carseats)
        summary(tree.carseats)
        plot(tree.carseats)
        text(tree.carseats,pretty=0)
        tree.carseats
        set.seed(2)
        train=sample(1:nrow(Carseats), 200)
        Carseats.test=Carseats[-train,]
        High.test=High[-train]
        tree.carseats=tree(High~.-Sales,Carseats,subset=train)
        tree.pred=predict(tree.carseats,Carseats.test,type="class")
        table(tree.pred,High.test)
        (86+57)/200
        set.seed(3)
        cv.carseats=cv.tree(tree.carseats,FUN=prune.misclass)
        names(cv.carseats)
        cv.carseats
        par(mfrow=c(1,2))
        plot(cv.carseats$size,cv.carseats$dev,type="b")
        plot(cv.carseats$k,cv.carseats$dev,type="b")
        prune.carseats=prune.misclass(tree.carseats,best=9)
        plot(prune.carseats)
        text(prune.carseats,pretty=0)
        tree.pred=predict(prune.carseats,Carseats.test,type="class")
        table(tree.pred,High.test)
        (94+60)/200
        prune.carseats=prune.misclass(tree.carseats,best=15)
        plot(prune.carseats)
        text(prune.carseats,pretty=0)
        tree.pred=predict(prune.carseats,Carseats.test,type="class")
        table(tree.pred,High.test)
        (86+62)/200
        
        # Fitting Regression Trees
        
        library(MASS)
        set.seed(1)
        train = sample(1:nrow(Boston), nrow(Boston)/2)
        tree.boston=tree(medv~.,Boston,subset=train)
        summary(tree.boston)
        plot(tree.boston)
        text(tree.boston,pretty=0)
        cv.boston=cv.tree(tree.boston)
        plot(cv.boston$size,cv.boston$dev,type='b')
        prune.boston=prune.tree(tree.boston,best=5)
        plot(prune.boston)
        text(prune.boston,pretty=0)
        yhat=predict(tree.boston,newdata=Boston[-train,])
        boston.test=Boston[-train,"medv"]
        plot(yhat,boston.test)
        abline(0,1)
        mean((yhat-boston.test)^2)
        
        # Bagging and Random Forests
        
        library(randomForest)
        set.seed(1)
        bag.boston=randomForest(medv~.,data=Boston,subset=train,mtry=13,importance=TRUE)
        bag.boston
        yhat.bag = predict(bag.boston,newdata=Boston[-train,])
        plot(yhat.bag, boston.test)
        abline(0,1)
        mean((yhat.bag-boston.test)^2)
        bag.boston=randomForest(medv~.,data=Boston,subset=train,mtry=13,ntree=25)
        yhat.bag = predict(bag.boston,newdata=Boston[-train,])
        mean((yhat.bag-boston.test)^2)
        set.seed(1)
        rf.boston=randomForest(medv~.,data=Boston,subset=train,mtry=6,importance=TRUE)
        yhat.rf = predict(rf.boston,newdata=Boston[-train,])
        mean((yhat.rf-boston.test)^2)
        importance(rf.boston)
        varImpPlot(rf.boston)
        
        # Boosting
        
        library(gbm)
        set.seed(1)
        boost.boston=gbm(medv~.,data=Boston[train,],distribution="gaussian",n.trees=5000,interaction.depth=4)
        summary(boost.boston)
        par(mfrow=c(1,2))
        plot(boost.boston,i="rm")
        plot(boost.boston,i="lstat")
        yhat.boost=predict(boost.boston,newdata=Boston[-train,],n.trees=5000)
        mean((yhat.boost-boston.test)^2)
        boost.boston=gbm(medv~.,data=Boston[train,],distribution="gaussian",n.trees=5000,interaction.depth=4,shrinkage=0.2,verbose=F)
        yhat.boost=predict(boost.boston,newdata=Boston[-train,],n.trees=5000)
        mean((yhat.boost-boston.test)^2)
        
        
        
        # Chapter 9 Lab: Support Vector Machines
        
        # Support Vector Classifier
        
        set.seed(1)
        x=matrix(rnorm(20*2), ncol=2)
        y=c(rep(-1,10), rep(1,10))
        x[y==1,]=x[y==1,] + 1
        plot(x, col=(3-y))
        dat=data.frame(x=x, y=as.factor(y))
        library(e1071)
        svmfit=svm(y~., data=dat, kernel="linear", cost=10,scale=FALSE)
        plot(svmfit, dat)
        svmfit$index
        summary(svmfit)
        svmfit=svm(y~., data=dat, kernel="linear", cost=0.1,scale=FALSE)
        plot(svmfit, dat)
        svmfit$index
        set.seed(1)
        tune.out=tune(svm,y~.,data=dat,kernel="linear",ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100)))
        summary(tune.out)
        bestmod=tune.out$best.model
        summary(bestmod)
        xtest=matrix(rnorm(20*2), ncol=2)
        ytest=sample(c(-1,1), 20, rep=TRUE)
        xtest[ytest==1,]=xtest[ytest==1,] + 1
        testdat=data.frame(x=xtest, y=as.factor(ytest))
        ypred=predict(bestmod,testdat)
        table(predict=ypred, truth=testdat$y)
        svmfit=svm(y~., data=dat, kernel="linear", cost=.01,scale=FALSE)
        ypred=predict(svmfit,testdat)
        table(predict=ypred, truth=testdat$y)
        x[y==1,]=x[y==1,]+0.5
        plot(x, col=(y+5)/2, pch=19)
        dat=data.frame(x=x,y=as.factor(y))
        svmfit=svm(y~., data=dat, kernel="linear", cost=1e5)
        summary(svmfit)
        plot(svmfit, dat)
        svmfit=svm(y~., data=dat, kernel="linear", cost=1)
        summary(svmfit)
        plot(svmfit,dat)
        
        # Support Vector Machine
        
        set.seed(1)
        x=matrix(rnorm(200*2), ncol=2)
        x[1:100,]=x[1:100,]+2
        x[101:150,]=x[101:150,]-2
        y=c(rep(1,150),rep(2,50))
        dat=data.frame(x=x,y=as.factor(y))
        plot(x, col=y)
        train=sample(200,100)
        svmfit=svm(y~., data=dat[train,], kernel="radial",  gamma=1, cost=1)
        plot(svmfit, dat[train,])
        summary(svmfit)
        svmfit=svm(y~., data=dat[train,], kernel="radial",gamma=1,cost=1e5)
        plot(svmfit,dat[train,])
        set.seed(1)
        tune.out=tune(svm, y~., data=dat[train,], kernel="radial", ranges=list(cost=c(0.1,1,10,100,1000),gamma=c(0.5,1,2,3,4)))
        summary(tune.out)
        table(true=dat[-train,"y"], pred=predict(tune.out$best.model,newx=dat[-train,]))
        
        # ROC Curves
        
        library(ROCR)
        rocplot=function(pred, truth, ...){
          predob = prediction(pred, truth)
          perf = performance(predob, "tpr", "fpr")
          plot(perf,...)}
        svmfit.opt=svm(y~., data=dat[train,], kernel="radial",gamma=2, cost=1,decision.values=T)
        fitted=attributes(predict(svmfit.opt,dat[train,],decision.values=TRUE))$decision.values
        par(mfrow=c(1,2))
        rocplot(fitted,dat[train,"y"],main="Training Data")
        svmfit.flex=svm(y~., data=dat[train,], kernel="radial",gamma=50, cost=1, decision.values=T)
        fitted=attributes(predict(svmfit.flex,dat[train,],decision.values=T))$decision.values
        rocplot(fitted,dat[train,"y"],add=T,col="red")
        fitted=attributes(predict(svmfit.opt,dat[-train,],decision.values=T))$decision.values
        rocplot(fitted,dat[-train,"y"],main="Test Data")
        fitted=attributes(predict(svmfit.flex,dat[-train,],decision.values=T))$decision.values
        rocplot(fitted,dat[-train,"y"],add=T,col="red")
        
        # SVM with Multiple Classes
        
        set.seed(1)
        x=rbind(x, matrix(rnorm(50*2), ncol=2))
        y=c(y, rep(0,50))
        x[y==0,2]=x[y==0,2]+2
        dat=data.frame(x=x, y=as.factor(y))
        par(mfrow=c(1,1))
        plot(x,col=(y+1))
        svmfit=svm(y~., data=dat, kernel="radial", cost=10, gamma=1)
        plot(svmfit, dat)
        
        # Application to Gene Expression Data
        
        library(ISLR)
        names(Khan)
        dim(Khan$xtrain)
        dim(Khan$xtest)
        length(Khan$ytrain)
        length(Khan$ytest)
        table(Khan$ytrain)
        table(Khan$ytest)
        dat=data.frame(x=Khan$xtrain, y=as.factor(Khan$ytrain))
        out=svm(y~., data=dat, kernel="linear",cost=10)
        summary(out)
        table(out$fitted, dat$y)
        dat.te=data.frame(x=Khan$xtest, y=as.factor(Khan$ytest))
        pred.te=predict(out, newdata=dat.te)
        table(pred.te, dat.te$y)
        
        
        
        # Chapter 10 Lab 1: Principal Components Analysis
        
        states=row.names(USArrests)
        states
        names(USArrests)
        apply(USArrests, 2, mean)
        apply(USArrests, 2, var)
        pr.out=prcomp(USArrests, scale=TRUE)
        names(pr.out)
        pr.out$center
        pr.out$scale
        pr.out$rotation
        dim(pr.out$x)
        biplot(pr.out, scale=0)
        pr.out$rotation=-pr.out$rotation
        pr.out$x=-pr.out$x
        biplot(pr.out, scale=0)
        pr.out$sdev
        pr.var=pr.out$sdev^2
        pr.var
        pve=pr.var/sum(pr.var)
        pve
        plot(pve, xlab="Principal Component", ylab="Proportion of Variance Explained", ylim=c(0,1),type='b')
        plot(cumsum(pve), xlab="Principal Component", ylab="Cumulative Proportion of Variance Explained", ylim=c(0,1),type='b')
        a=c(1,2,8,-3)
        cumsum(a)
        
        
        # Chapter 10 Lab 2: Clustering
        
        # K-Means Clustering
        
        set.seed(2)
        x=matrix(rnorm(50*2), ncol=2)
        x[1:25,1]=x[1:25,1]+3
        x[1:25,2]=x[1:25,2]-4
        km.out=kmeans(x,2,nstart=20)
        km.out$cluster
        plot(x, col=(km.out$cluster+1), main="K-Means Clustering Results with K=2", xlab="", ylab="", pch=20, cex=2)
        set.seed(4)
        km.out=kmeans(x,3,nstart=20)
        km.out
        plot(x, col=(km.out$cluster+1), main="K-Means Clustering Results with K=3", xlab="", ylab="", pch=20, cex=2)
        set.seed(3)
        km.out=kmeans(x,3,nstart=1)
        km.out$tot.withinss
        km.out=kmeans(x,3,nstart=20)
        km.out$tot.withinss
        
        # Hierarchical Clustering
        
        hc.complete=hclust(dist(x), method="complete")
        hc.average=hclust(dist(x), method="average")
        hc.single=hclust(dist(x), method="single")
        par(mfrow=c(1,3))
        plot(hc.complete,main="Complete Linkage", xlab="", sub="", cex=.9)
        plot(hc.average, main="Average Linkage", xlab="", sub="", cex=.9)
        plot(hc.single, main="Single Linkage", xlab="", sub="", cex=.9)
        cutree(hc.complete, 2)
        cutree(hc.average, 2)
        cutree(hc.single, 2)
        cutree(hc.single, 4)
        xsc=scale(x)
        plot(hclust(dist(xsc), method="complete"), main="Hierarchical Clustering with Scaled Features")
        x=matrix(rnorm(30*3), ncol=3)
        dd=as.dist(1-cor(t(x)))
        plot(hclust(dd, method="complete"), main="Complete Linkage with Correlation-Based Distance", xlab="", sub="")
        
        
        # Chapter 10 Lab 3: NCI60 Data Example
        
        # The NCI60 data
        
        library(ISLR)
        nci.labs=NCI60$labs
        nci.data=NCI60$data
        dim(nci.data)
        nci.labs[1:4]
        table(nci.labs)
        
        # PCA on the NCI60 Data
        
        pr.out=prcomp(nci.data, scale=TRUE)
        Cols=function(vec){
          cols=rainbow(length(unique(vec)))
          return(cols[as.numeric(as.factor(vec))])
        }
        par(mfrow=c(1,2))
        plot(pr.out$x[,1:2], col=Cols(nci.labs), pch=19,xlab="Z1",ylab="Z2")
        plot(pr.out$x[,c(1,3)], col=Cols(nci.labs), pch=19,xlab="Z1",ylab="Z3")
        summary(pr.out)
        plot(pr.out)
        pve=100*pr.out$sdev^2/sum(pr.out$sdev^2)
        par(mfrow=c(1,2))
        plot(pve,  type="o", ylab="PVE", xlab="Principal Component", col="blue")
        plot(cumsum(pve), type="o", ylab="Cumulative PVE", xlab="Principal Component", col="brown3")
        
        # Clustering the Observations of the NCI60 Data
        
        sd.data=scale(nci.data)
        par(mfrow=c(1,3))
        data.dist=dist(sd.data)
        plot(hclust(data.dist), labels=nci.labs, main="Complete Linkage", xlab="", sub="",ylab="")
        plot(hclust(data.dist, method="average"), labels=nci.labs, main="Average Linkage", xlab="", sub="",ylab="")
        plot(hclust(data.dist, method="single"), labels=nci.labs,  main="Single Linkage", xlab="", sub="",ylab="")
        hc.out=hclust(dist(sd.data))
        hc.clusters=cutree(hc.out,4)
        table(hc.clusters,nci.labs)
        par(mfrow=c(1,1))
        plot(hc.out, labels=nci.labs)
        abline(h=139, col="red")
        hc.out
        set.seed(2)
        km.out=kmeans(sd.data, 4, nstart=20)
        km.clusters=km.out$cluster
        table(km.clusters,hc.clusters)
        hc.out=hclust(dist(pr.out$x[,1:5]))
        plot(hc.out, labels=nci.labs, main="Hier. Clust. on First Five Score Vectors")
        table(cutree(hc.out,4), nci.labs)
        
        