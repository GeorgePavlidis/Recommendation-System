# demo.py ------------------------------------------------------------------------------------------
# 
# 
#
# --------------------------------------------------------------------------------------------------
from io import StringIO
import html
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import jaccard_similarity_score
from sklearn.feature_extraction.text import CountVectorizer
import warnings


class RecommendationSystem:

   def __init__(self):
      warnings.filterwarnings('ignore')
      print("<B>Do your magic things\n")

   @property
   def dataPreProcessing(self):
      # ============================================================================================
      # Read csv files
      # ============================================================================================
      print("<B>Read csv files...")
      # Ratings
      # Read Ratings and store them in a dataframe
      bookRating = pd.read_csv("./bx/BX-Book-Ratings.csv",
                               sep=';',
                               encoding='ISO-8859-1',
                               dtype=object)
      bookRating = bookRating.dropna()
      print("<B>Ratings loaded")

      # Books
      # It is necessary to trasform HTML items such as &amp; in order to read properly the csv file
      with open("./bx/BX-Books.csv", 'r', encoding='ISO-8859-1') as file:
         content = html.unescape(file.read())
      # Read Books and store them in a dataframe
      books = pd.read_csv(StringIO(content),
                          sep=';',
                          encoding='ISO-8859-1',
                          error_bad_lines=False,
                          dtype=object)

      # Remove from dataframe columns of urls. Keep only the necessary columns.
      books = books.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1)
      books = books.dropna()

      print("<B>Books loaded")

      # Users
      # Read Users and store them in a dataframe
      users = pd.read_csv("./bx/BX-Users.csv",
                          sep=';',
                          encoding='ISO-8859-1',
                          dtype=object)
      users = users.dropna()
      print("<B>Users loaded")

      # Drop not valid values
      print("<B>All data are loaded\n\n")

      # ============================================================================================
      # Remove users with less than 5 ratings and books with less than 10 ratings
      # ============================================================================================
      # 1st step: From dataframe user-book-rating remove not valid users and books
      # 2nd step: Mask users and books dataframes, keep only valids lines
      print("<B>Remove not valid lines from bookRating...")

      # Remove not valid ratings
      bookRating = bookRating[bookRating['User-ID'].isin(users['User-ID'])]
      bookRating = bookRating[bookRating['ISBN'].isin(books['ISBN'])]
      bookRating = bookRating[bookRating['Book-Rating'] != str(0)]

      # Remove books with less than 10 ratings
      validBooks = bookRating['ISBN'].value_counts()
      validBooks = validBooks[validBooks >= 10].index
      bookRating = bookRating[bookRating['ISBN'].isin(validBooks)]
      print("<B>Books with less than 10 ratings removed")

      # Remove users with less than 5 ratings
      validUsers = bookRating['User-ID'].value_counts()
      validUsers = validUsers[validUsers >= 5].index
      bookRating = bookRating[bookRating['User-ID'].isin(validUsers)][:1000]
      print("<B>Users with less than 5 ratings removed")

      # Remove not valid books
      # if book-ID (ISBN) is not in the rating list then drop it
      books = books[books['ISBN'].isin(bookRating['ISBN'])]
      books = books[books['Year-Of-Publication'].astype(int) != 0]
      print("<B>Clear book dataframe from not valid lines")

      # Remove not valid users
      # if user-ID is not in the rating list then drop it
      users = users[users['User-ID'].isin(bookRating['User-ID'])]
      print("<B>Clear user dataframe from not valid lines\n\n")

      # ============================================================================================
      # Books titles text pre-processing
      # ============================================================================================
      # 1st step: tokenize and remove punctuation( '-', '!', '?', etc)
      # 2nd step: Apply stemming
      # 3rd step: remove stop words
      # 4th step: Vectorize, create a vector for each title

      print("<B>Pre-process the books titles...")

      # Initialize necessary tools
      print("<B>Initialize...")
      tokenizer = RegexpTokenizer(r'\w+')
      snowball = SnowballStemmer(language='english')
      stop_words = np.array(stopwords.words('english'), dtype='<U16')

      # Apply pre-processing
      print("<B>Apply tokenizer, stemming, remove stop words...")
      titles = books['Book-Title']
      titles = books['Book-Title'].map(tokenizer.tokenize)
      titles = titles.apply(self._stemming_stop_words, args=(snowball, stop_words,))
      books['Title'] = titles

      # Vectorize titles
      print("<B>Vectorize...")
      vectorizer = CountVectorizer()
      vocab = vectorizer.fit_transform(titles)

      print("<B>Pre-process is done\n\n")

      return books, users, bookRating, vectorizer, vocab

   def _stemming_stop_words(self, sentence, snowball, stop_words):
      """
         This function perform stemming and remove stop words
      """
      # Check if any stem_words exits in stop_words set, if yes remove this word, else keep it
      sentence = np.array(sentence)[np.logical_not(np.isin(sentence, stop_words))]
      # Stem the words of th e sentence (e.g is, are, was, were --> be)
      stem_words = [snowball.stem(word) for word in sentence]

      # return stem_words
      return ' '.join(stem_words)

   def userKeyWordsPool(self, bookRating, books):
      """
      This function returns information about the most most Valued book for each user
         (User-ID, Title, Book-Author, Year-Of-Publication
      """
      # Find the 3 Most Valued (Highest rate) books for each user.
      mostValued = bookRating.sort_values(['User-ID', 'Book-Rating'], ascending=False)
      mostValued = mostValued.groupby('User-ID')
      mostValued = mostValued.head(3)

      # Create a word pool from the most Valued book for each user.
      # Also, keep the Title, Author and Year of Publication of thess books.
      # These word pool will be the signature of a user, based on wich will work the
      # recommendation system.
      userKeyWordsPool = mostValued.set_index('ISBN')
      userKeyWordsPool = userKeyWordsPool.join(books.set_index('ISBN'), how='inner')
      userKeyWordsPool = userKeyWordsPool[['User-ID', 'Title',
                                           'Book-Author', 'Year-Of-Publication']]

      return userKeyWordsPool

   def recommendation(self, userKeyWordsPool, vectorizer, vocab, users, bookRating,
                      books, method='Jaccard'):
      """
      This function returns 10 items from <books> for everu user in <userKeyWordsPool>. This 10
      items are the most recommented for a paricular user based on <userKeyWordsPool>.
      """

      # Set weights based on the method of calculation
      if method == 'Dice':
         weightTitle = 0.5
         weightAuthor = 0.3
         weightRealiseDate = 0.2
      elif method == 'Jaccard':
         weightTitle = 0.2
         weightAuthor = 0.4
         weightRealiseDate = 0.4
      else:
         print("Invalid method. Jaccard will be used instead.")
         weightTitle = 0.2
         weightAuthor = 0.4
         weightRealiseDate = 0.4

      print("<B>Find recommendations with ", method, "...")
      recommendations = []
      for user in users:

         print("<B>User", user, "...", end='')

         # =========================================================================================
         # Extract users signature
         # =========================================================================================
         # Get the words pool of particular user
         wordsPool = userKeyWordsPool[userKeyWordsPool['User-ID'] == user]
         wordsPool = wordsPool.drop(['User-ID'], axis=1)

         # Join all the title key words together
         title = ' '.join(np.array(wordsPool['Title']))
         title = vectorizer.transform([title, ])
         # Join the author and  years of Publication
         author = np.array(wordsPool['Book-Author'])
         realiseDate = np.array(wordsPool['Year-Of-Publication'], dtype=int)

         # =========================================================================================
         # Calculate the Score
         # =========================================================================================
         # 1st Step: Authors contribution to final score
         books['Score'] = weightAuthor * books['Book-Author'].isin(author)

         # 2nd Step: Year-Of-Publication contribution to final score
         books['Score'] = books['Score'] + weightRealiseDate * books['Year-Of-Publication'] \
            .astype(int).apply(self._minDateDistance, args=(realiseDate,))

         # 3rd Step: Title contribution to final score
         if method == 'Dice':
            books['Score'] = books['Score'] + \
                             weightTitle * self._dice_coefficient(vocab.toarray(), title.toarray())
         else:
            books['Score'] = books['Score'] + \
                             weightTitle * self._jaccard_similarity(vocab.toarray(),
                                                                    title.toarray())

         # 4th Step: Zeroing the scores of book that user have rated
         mask = bookRating[bookRating['User-ID']==user]['ISBN']
         books['Score'][books['ISBN'].isin(mask)] = 0.0

         # 5th Step: Sort books by score and return the first 10
         recommendations.append(books[['ISBN', 'Score']].sort_values(by=['Score'],
                                                                     ascending=False).head(10))
         print("Complete")
      print('\n\n')
      return recommendations

   def _minDateDistance(self, year, realiseDate):
      """
      This is a private function that calculates the min absolute normalized distance
      between of Publication Years. The method of calculation 1-(|year-realiseDate|/2005)
      """

      aHigherPossibleValue = 2005
      absDiff = np.absolute(year - realiseDate)
      normDiff = np.divide(absDiff, aHigherPossibleValue)
      minDiff = np.min(normDiff)

      return 1 - minDiff

   def _jaccard_similarity(self, vocabs, title):
      """
      This is a private function that calculates Jaccard similarity coefficient score between a
      Nd-array and 1d-array. For every item of Nd-array call the jaccard_similarity_score and
      output the similarity with the 1d-array.
      Returns:
         Nd-array same dimensionality like vocab
      """

      scores = np.zeros([vocabs.shape[0], ], dtype=float)
      for item in range(vocabs.shape[0]):
         scores[item] = jaccard_similarity_score(vocabs[item].reshape(-1, ),
                                                 title.reshape(-1, ))

      return scores

   def _dice_coefficient(self, vocabs, title):
      """
      This is a private function that calculates dice coefficient score between a Nd-array and
      1d-array. For every item of Nd-array call calculate dice coefficient according to formula
      [ 2(a * b).sum() /a.sum() +b.sum() ] and output the similarity with the 1d-array.
      """

      scores = np.zeros([vocabs.shape[0], ], dtype=float)

      for item in range(vocabs.shape[0]):
         titles = vocabs[item].reshape(-1, )
         theTitle = title.reshape(-1, )
         intersection = np.multiply(titles, theTitle)
         scores[item] = np.divide(2. * intersection.sum(), titles.sum() + theTitle.sum())

      return scores
