url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/ann-train.data"
thyroid_train <- read.csv(file = url, stringsAsFactors = FALSE, sep = "", header = FALSE)
names(thyroid_train) <- c('Age', 'Sex', 'On_thyroxine', 'Query_on_thyroxine', 'On_antithyroid_medication',
                    'Sick', 'Pregnant', 'Thyroid_surgery', 'I131_treatment', 'Query_hypothyroid', 
                    'Query_hyperthyroid', 'Lithium', 'Goitre', 'Tumor', 'Hypopituitary', 
                    'Psych', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'Class')
#write.csv(thyroid_train, "thyroid-train.csv")

url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/ann-test.data"
thyroid_test <- read.csv(file = url, stringsAsFactors = FALSE, sep = "", header = FALSE)
names(thyroid_test) <- c('Age', 'Sex', 'On_thyroxine', 'Query_on_thyroxine', 'On_antithyroid_medication',
                    'Sick', 'Pregnant', 'Thyroid_surgery', 'I131_treatment', 'Query_hypothyroid', 
                    'Query_hyperthyroid', 'Lithium', 'Goitre', 'Tumor', 'Hypopituitary', 
                    'Psych', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'Class')

#write.csv(thyroid_test, "thyroid-test.csv")

all_together <- rbind(thyroid_train, thyroid_test)
write.csv(all_together, "thyroid-complete.csv")