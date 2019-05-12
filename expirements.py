import sys


if __name__ == "__main__":
	try:
		if not len(sys.argv)==7 :
			raise
	except IndexError:
		print ("expirements.py <output-file> <T> <N> <M> <X> <K>")
	except:
		print ("lol ")

		sys.exit(2)
	exportF = sys.argv[1]
	export_file=open(exportF,"w")
	for i in range(2,7):
		export_file.write(sys.argv[i]+'\n')
	export_file.close()