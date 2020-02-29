# recommendationExecution.py -----------------------------------------------------------------------
#
#
#
# --------------------------------------------------------------------------------------------------
import recommendationSystem as RS
import numpy as np
import os
RS = RS.RecommendationSystem()

# Read Data
books, users, bookRating, vectorizer, vocab = RS.dataPreProcessing
# Create User signature based on their rating
usrsKeyWordPool = RS.userKeyWordsPool(bookRating, books)
# Choose 5 random users
usersForRecom = np.array(users.sample(5)['User-ID'])

# Produce Jaccard recommendation for these users
recommendationJaccard = RS.recommendation(usrsKeyWordPool, vectorizer, vocab,
                                          usersForRecom, bookRating,
                                          books, method='Jaccard')
# Produce Dice recommendation for these users
recommendationDice = RS.recommendation(usrsKeyWordPool, vectorizer, vocab,
                                       usersForRecom, bookRating,
                                       books, method='Dice')


for index in range(0, len(recommendationDice)):

   print('<B>User', usersForRecom[index], ':')
   print('<B>Recommendations Jaccard: \n')
   print(recommendationJaccard[index])
   print('\n<B>Recommendations Dice: ')
   print(recommendationDice[index])

   # ===============================================================================================
   # Export Recommendations
   # ===============================================================================================
   if not os.path.exists("./output"):
      os.makedirs("./output")

   fileNameJaccard = "./output/" + "User-" + usersForRecom[index] + '-Jaccard.csv'
   fileNameDice = "./output/" + "User-" + usersForRecom[index] + '-Dice.csv'

   recommendationJaccard[index].to_csv(fileNameJaccard, index = None, header=True)
   recommendationDice[index].to_csv(fileNameDice, index = False, header=True)

   # ===============================================================================================
   # Overlap
   # ===============================================================================================
   # Find the overlap between two recommendation lists
   JaccardDiceOverlap = np.array(np.array(recommendationJaccard[index]['ISBN']) == np.array(
      recommendationDice[index]['ISBN']), dtype=int).mean()

   # ===============================================================================================
   # Golden Standard
   # ===============================================================================================
   print ('   <B>Jaccard-Dice Overlap:', JaccardDiceOverlap)
   #Compine two recommendation lists
   allResults = recommendationJaccard[index].append(recommendationDice[index])
   # Calculate the Frequency of each book
   allResults['Frequency'] = allResults['ISBN'].map(allResults['ISBN'].value_counts())
   # Keep the unique rows and sort them first by Frequency and then by score in order to compute
   # the golden Standard recommendation list
   allResults = allResults.groupby('ISBN').max().reset_index()\
                          .sort_values(by=['Frequency','Score'],ascending=False).head(10)
   goldenStandard = np.array(allResults['ISBN'])
   print('<B>Recommendations goldenStandard: \n')
   print(allResults)
   # ===============================================================================================
   # Overlap
   # ===============================================================================================
   # Find the overlap between  golden Standard and Jaccard recommendation
   JaccardOverlap = np.array(np.array(recommendationJaccard[index]['ISBN']) == goldenStandard,
                             dtype=int).mean()
   # Find the overlap between  golden Standard and Jaccard recommendation
   DiceOverlap = np.array(np.array(recommendationDice[index]['ISBN']) == goldenStandard,
                          dtype=int).mean()

   print ('   <B>Jaccard-Standard Overlap:', JaccardOverlap)
   print ('   <B>Dice-Standard Overlap:', DiceOverlap,'\n\n')


