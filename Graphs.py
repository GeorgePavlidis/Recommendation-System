import os
import sys
import matplotlib.pyplot as plt
import numpy as np

def find_mean(dir,files):
	list=[]
	for file in files:
		with open(dir+'/'+file,"r+") as import_file:
			import_file.readline()
			list.append(float(import_file.readline()))
	return reduce(lambda x, y: x + y, list) / len(list)



if __name__ == "__main__":
	try:
		if not len(sys.argv)==3 :
			raise
		path=sys.argv[1]

	except IndexError:
		print ("demo.py <path> <categories>(example: 'k=3 k=5 k=10')")
	except:
		print ( len(sys.argv))

		sys.exit(2)

	i=0
	jaccard_user_mean=[]
	jaccard_obj_mean=[]

	cosine_user_mean=[]
	cosine_obj_mean=[]

	pearson_user_mean=[]
	pearson_obj_mean=[]
	##
	#find all the files with mean_absolute_error
	# and calculate their mean 
	##
	for dir in os.walk(path):
		#print dir[2]
		jaccard_user = [x for x in dir[2] if x.startswith('mean_absolute_error_jaccard_user')]
		if jaccard_user != []:
			jaccard_user_mean.append(find_mean(dir[0],jaccard_user))

		jaccard_object = [x for x in dir[2] if x.startswith('mean_absolute_error_jaccard_obj')]
		if jaccard_object != []:
			jaccard_obj_mean.append(find_mean(dir[0],jaccard_object))

		cosine_user = [x for x in dir[2] if x.startswith('mean_absolute_error_cosine_user')]
		if cosine_user != []:
			cosine_user_mean.append(find_mean(dir[0],cosine_user))

		cosine_object = [x for x in dir[2] if x.startswith('mean_absolute_error_cosine_obj')]
		if cosine_object != []:
			cosine_obj_mean.append(find_mean(dir[0],cosine_object))


		pearson_user = [x for x in dir[2] if x.startswith('mean_absolute_error_pearson_user')]
		if pearson_user != []:
			pearson_user_mean.append(find_mean(dir[0],pearson_user))

		pearson_object = [x for x in dir[2] if x.startswith('mean_absolute_error_pearson_obj')]
		if pearson_object != []:
			pearson_obj_mean.append(find_mean(dir[0],pearson_object))

	print 'jaccard_user_mean',jaccard_user_mean
	print 'jaccard_obj_mean',jaccard_obj_mean

	print 'cosine_user_mean',cosine_user_mean
	print 'cosine_obj_mean',cosine_obj_mean

	print 'pearson_user_mean',pearson_user_mean
	print 'pearson_obj_mean',pearson_obj_mean

	x_axis = sys.argv[2].split(' ')
	print x_axis




	#Create the Graphs

	index = np.arange(len(x_axis))

	fig, axes = plt.subplots(nrows=2, ncols=1)
	ax0, ax1 = axes.flatten()

	bar_width = 0.15

	opacity = 0.80
	error_config = {'ecolor': '0.3'}
	rects1 = ax0.bar(index, jaccard_user_mean, bar_width,
					alpha=opacity, color='b',
					label='jaccard')

	rects2 = ax0.bar(index + bar_width, cosine_user_mean, bar_width, 
					alpha=opacity, color='r',
					label='cosine')

	rects3 = ax0.bar(index - bar_width, pearson_user_mean, bar_width,
					alpha=opacity, color='g',
					label='pearson')

	rects4 = ax1.bar(index, jaccard_user_mean, bar_width,
					alpha=opacity, color='b',
					label='jaccard')

	rects5 = ax1.bar(index + bar_width, cosine_user_mean, bar_width,
					alpha=opacity, color='r',
					label='cosine')

	rects6 = ax1.bar(index - bar_width, pearson_user_mean, bar_width,
					alpha=opacity, color='g',
					label='pearson')

	ax0.set_title('User-to-user')
	ax0.set_xticks(index + bar_width / 3)
	ax0.set_xticklabels(x_axis)
	ax0.set_ylim(bottom=None, top=1.0)
	ax1.set_title('Object-to-object')
	ax1.set_xticks(index + bar_width / 3)
	ax1.set_xticklabels(x_axis)
	ax1.set_ylim(bottom=None, top=1.0)



	ax0.legend()
	ax1.legend()

	fig.tight_layout()
	plt.savefig(path+'/plot.png')