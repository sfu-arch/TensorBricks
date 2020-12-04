import sys, pprint, re, operator

# Colorbrewer divergent
# color_palette = ['#d7191c','#fdae61','#ffffbf', \
	# '#abdda4','#2b83ba', '#acacac']

# Tableau 10 palette (reordered)
color_palette = [  \
	# Blue
	'#4e79a7', \
	# Red
	# '#c15759', \
        '#ab6267', \
	# Cyan
	# '#76b7b2', \
        '#48a3ba', \
	# Green
	'#59a14f', \
	# Purple
	'#b07aa1', \
	# Orange
	'#f28e2b', \
	# Yellow
	'#edc948', \
	'#ff9da7', \
	'#9c755f', \
	'#bab0ac']

# Tableau 10 palette (original)
# color_palette = [  \
	# '#4e79a7', \
	# '#f28e2b', \
	# '#c15759', \
	# '#76b7b2', \
	# '#ff9da7', \
	# '#9c755f', \
	# '#59a14f', \
	# '#edc948', \
	# '#b07aa1', \
	# '#bab0ac']


def getcolor(idx):
    global color_palette
    return color_palette[idx]

def getpalette():
    global color_palette
    return color_palette

def rename(names):
    regex = re.compile("^\d+\.")
    remove_numbers = [regex.sub('', s) if s.strip() not in ('181.mcf', '429.mcf') else s for s in names ]
    truncated = [s[:7] for s in remove_numbers]
    # pp = pprint.PrettyPrinter(indent=2)
    # pp.pprint(truncated)
    return truncated

def gm(iterable):
    return (reduce(operator.mul, iterable)) ** (1.0/len(iterable))




def calc_columnwise_percent(A):
    #transpose
    A=[list(x) for x in zip(*A)]
    #convert to floats
    A= [[float(x) for x in row] for row in A]
    #Calculate %breakdown
    A=[[x*100/sum(row) for x in row] for row in A]
    #transpose back to original
    A=[list(x) for x in zip(*A)]
    return A



if __name__ == "__main__":
    rename([sys.argv[1]]*5)
