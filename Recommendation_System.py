# aporia : 
# 1? stis periptwseis pou den exei va8mologi8ei na valw 0 ?
# 2? poia sunarthsh na xris, ama den exei sumplirwsei oute o allos 
# 3? einai swstos o tropos sumplirwseis twn newn pinakwn 
# 4? oi omoiothtes vgenoun swsta ?
# 5?  grafikes parastaseis



import sys
import os as os
import  itertools
import numpy as np
from numpy import corrcoef
import scipy.sparse as sc
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from scipy.stats.stats import pearsonr
from sklearn.neighbors import NearestNeighbors 
from sklearn.metrics import mean_absolute_error

#jaccard_similarity_score(y_true, y_pred)
#sklearn.metrics.pairwise.cosine_similarity(X, Y=None, dense_output=True)
#nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
#distances, indices = nbrs.kneighbors(X)
#numpy.corrcoef(list1, list2) or pearsonr(x,y)
#mean_absolute_error(y_true, y_pred, multioutput='raw_values')

#T: o arithmos epanalipsewn ths ekteleshs 
T = 0
#N: to plhthos twn xrhstwn toy systhmatos
N = 0
#M: to plhthos twn antikeimenwn toy systhmatos
M = 0 
#X: to pososto pou deixnei poso gematos einai o pinakas
X = 0 
#K: to plithws twn kontinwterwn geitonwn gia thn problepsh
K = 0
#dir pou 8a apo8ekeutoyn ta apotelesmta
dir=''

def readData(fileName):
	#
	#Read the input file and insert the values to variables
	#
	global T, N, M, X, K
	with open(fileName,"r+") as import_file:
		stack=[]

		T = int(import_file.readline())
		N = int(import_file.readline())
		M = int(import_file.readline())
		X = int(import_file.readline())
		K = int(import_file.readline())

def createMatrix(n,m,x):
	#
	#Creating of a matrix with normally distributed values from 1 to 5 and Sparse = x	
	#

	#matrix with random values from 1 to 5 normally distributed
	matrix = np.array(np.random.random_integers(1,5, size=(n,m)))

	#matrix of same shape with previous one, sparse of x% and values 0 or 1
	x = sc.rand(n, m, density=x/100.0, format='csr')
	x.data[:] = 1

	#multiply each item of two matrixes
	arr = matrix*x.toarray()
	#arr[arr == 0] = np.nan
	return arr


def knn(matrix):
	#
	# Nearest Neighbors 
	#
	nbrs = NearestNeighbors(n_neighbors=N, algorithm='ball_tree').fit(matrix)
 	distances, indices = nbrs.kneighbors(matrix)
 	return distances, indices

def jaccard(mat):
	#
	# Crearing the matrix of jaccard distances 
	#
	jaccard_matrix = np.ones((N, N)) # Filled with ones
 	i=0
 	j=1
 	matrix=np.copy(mat)
 	matrix[matrix != 0] = 1
 	for combination in itertools.combinations(matrix,2): # itertools.combinations(matrix,2)  --> all possible combinations of 2 rows
 		#execute jaccard distances for each row 
 		jaccard_matrix[i,j] = jaccard_similarity_score(combination[0], combination[1])
 		jaccard_matrix[j,i] = jaccard_similarity_score(combination[0], combination[1])

		j=j+1
		if j==N:
			i+=1
			j=i+1
		
	return jaccard_matrix

def algorithm(n,m,x,k,t,model='user'):

		# creating of initial matrix user-to-user or object-to-object
		if model is 'user':
			matrix = createMatrix(n,m,x)
		elif model is 'obj':
			mat = createMatrix(n,m,x)
			matrix = mat.T # object-to-object = user-to-userMatrix.T 
		else :
			return None

		#simularities
		jaccard_matrix = jaccard(matrix)
		cosine_matrix = cosine_similarity(matrix)
		pearson_matrix = corrcoef(matrix)


		#knn distances, indices
		jaccard_neib_dist, jaccard_neib_ind = knn(jaccard_matrix)
		cosine_neib_dist, cosine_neib_ind = knn(cosine_matrix)
		pearson_neib_dist, pearson_neib_ind = knn(pearson_matrix)

		#new matrixes
		new_matrix_jaccard = np.copy(matrix)
		new_matrix_cosine = np.copy(matrix)
		new_matrix_pearson = np.copy(matrix)
		for row in range(n):
			for column in range(m):
				if matrix[row,column]== 0.0:
					numerator= 0
					denomirator = 0
					k=1
					#find K neibs, whitch is not null and check if all possible neibs is null
					for neib in range(1,n):
						if jaccard_neib_ind[row,neib]==0:
							continue
						if neib>=n:
							if denomirator==0:
								denomirator=1
							continue
						if k==K:
							break
						k+=1
						
						#new value
						denomirator += jaccard_matrix[row,jaccard_neib_ind[row,neib]]
						numerator += jaccard_matrix[row,jaccard_neib_ind[row,neib]]*matrix[jaccard_neib_ind[row,neib],column]
					new_matrix_jaccard[row,column] = numerator/denomirator

					numerator= 0
					denomirator = 0
					k=1
					#find K neibs, whitch is not null and check if all possible neibs is null
					for neib in range(1,n):
						if cosine_neib_ind[row,neib]==0:
							continue
						if neib>=n:
							if denomirator==0:
								denomirator=1
							continue
						if k==K:
							break
						k+=1
						
						#new value
						denomirator += cosine_matrix[row,cosine_neib_ind[row,neib]]
						numerator += cosine_matrix[row,cosine_neib_ind[row,neib]]*matrix[cosine_neib_ind[row,neib],column]
					new_matrix_cosine[row,column] = numerator/denomirator

					numerator= 0
					denomirator = 0
					k=1
					#find K neibs, whitch is not null and check if all possible neibs is null
					for neib in range(1,n):
						if pearson_neib_ind[row,neib]==0:
							continue
						if neib>=n:
							if denomirator==0:
								denomirator=1
							continue
						if k==K:
							break
						k+=1
						
						#new value
						denomirator += pearson_matrix[row,pearson_neib_ind[row,neib]]
						numerator += pearson_matrix[row,pearson_neib_ind[row,neib]]*matrix[pearson_neib_ind[row,neib],column]
					new_matrix_pearson[row,column] = numerator/denomirator
		


		# mean_absolute_errors
		errL = mean_absolute_error(matrix, new_matrix_jaccard, multioutput='raw_values')
		err  = mean_absolute_error(matrix, new_matrix_jaccard)
		exportFile(new_matrix_jaccard,errL,err,'jaccard_'+model+'('+str(t+1)+')')

		errL = mean_absolute_error(matrix, new_matrix_cosine, multioutput='raw_values')
		err  = mean_absolute_error(matrix, new_matrix_cosine)
		exportFile(new_matrix_cosine,errL,err, 'cosine_'+model+'('+str(t+1)+')')

		errL = mean_absolute_error(matrix, new_matrix_pearson, multioutput='raw_values')
		err  = mean_absolute_error(matrix, new_matrix_pearson)
		exportFile(new_matrix_pearson,errL,err, 'pearson_'+model+'('+str(t+1)+')')




def exportFile(matrix, errL,err, sim):

	
	name1 = dir+'/matrix_'+sim
	name2 = dir+'/mean_absolute_error_'+sim


	export_file=open(name1,"w")
	for row in matrix:
		for item in row:
			export_file.write(str(item)+' ')
		export_file.write('\n')
	export_file.close()

	export_file=open(name2,"w")
	for item in errL:
		export_file.write(str(item)+' ')
	export_file.write('\n'+str(err))
	export_file.close()










if __name__ == "__main__":
	try:
		if not len(sys.argv)==2 :
			if not len(sys.argv)==3:
				raise
			dir=sys.argv[2]+'/'
	except IndexError:
		print ("demo.py <configuration-file> (optionally) <dir>")
		print('<configuration-file> structure')
		print('T')
		print('M')
		print('N')
		print('X')
		print('K')

	except:
		print ( len(sys.argv))

		sys.exit(2)
	readData(sys.argv[1])

	directory=dir+str(T)+'-'+str(N)+'-'+str(M)+'-'+str(X)+'-'+str(K)+'/'

	for i in range(T):
		dir=directory+'user-to-user'
		if not os.path.exists(dir):
			os.makedirs(dir)
		algorithm(N,M,X,K,i)
		dir=directory+'object-to-object'
		if not os.path.exists(dir):
			os.makedirs(dir)
		algorithm(N,M,X,K,i,'obj')





