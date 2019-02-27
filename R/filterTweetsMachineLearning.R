#' @title Classify needs based on machine learning
#' 
#' @description
#' \code{filterTweetsMachineLearning} classifies a list of Tweets as
#' needs based on the random forest machine learning algorithm
#'
#' @details 
#' This function uses a machine learning algorithm (random forest) to
#' classify needs based on their content. It needs a training data set
#' with classified needs (indicated by 0=not a need, 1=a need)
#' 
#' @param dataToClassify a dataframe containing the Tweet messages to classify
#' @param trainingData a dataframe containing Tweets messages with a given classification (0=not a need, 1=a need)
#'
#' @return a dataframe with classified data
#' 
#' @author Dorian Proksch <dorian.proksch@hhl.de>
#'
#' @importFrom RTextTools create_container train_model classify_model create_matrix
#' 
#' @export
#' 
#' @examples
#' data(NMTrainingData)
#' data(NMdataToClassify)
#' smallNMTrainingData <- rbind(NMTrainingData[1:75,], NMTrainingData[101:175,])
#' smallNMdataToClassify <- rbind(NMdataToClassify[1:10,], NMdataToClassify[101:110,])
#' results <- filterTweetsMachineLearning(smallNMdataToClassify, smallNMTrainingData)
#'

filterTweetsMachineLearning <- function (dataToClassify, trainingData){

	if (missing(dataToClassify))
		stop("'dataToClassify' is missing.")

	if (missing(trainingData))
		stop("'trainingData' is missing.")

	merged_data <- as.matrix(rbind(trainingData, dataToClassify))

	doc_matrix <- create_matrix(merged_data[,"Tweets"], language="english", removeNumbers=TRUE,
	stemWords=TRUE, removeSparseTerms=.998)

	container <- create_container(doc_matrix, merged_data[, "isNeed"], trainSize=1:nrow(trainingData), 
	testSize=(nrow(trainingData)+1):(nrow(trainingData)+nrow(dataToClassify)), virgin=FALSE)
	
	RF <- train_model(container,"RF")
	
	RF_CLASSIFY <- classify_model(container, RF)

	classification <- as.integer(as.integer(as.vector(RF_CLASSIFY$FORESTS_LABEL)))

	final_results <- cbind(dataToClassify[,"Tweets"], classification)
	colnames(final_results) <- c("Tweets", "isNeed")
	final_results <- as.matrix(final_results)

	return(final_results)	
}