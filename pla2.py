import pandas as pd
import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

import sys

def main():
    '''get the data and results files '''
    data_file = str(sys.argv[1])
    result_file = str(sys.argv[2])
    file1 = open(result_file, "w")

    '''get the training data X'''
    # X has f1', 'f2', 'label
    X = pd.read_csv(data_file, sep=',', names=['f1', 'f2', 'label'])
    X = pd.DataFrame(X, columns=['f1', 'f2', 'label'])
    X.insert(2 ,'ones', np.ones_like(X['f1']))
    
    '''Create vector of weights'''
    #[w1,w2,b]
    W = [0,0,0]
    W_old = [0,0,0]
    alpha = 1

    '''Convergence condition'''
    convergence = False

    while not convergence:

        for i in range(X.shape[0]):
            xi = X.loc[i,['f1','f2','ones']]
            yi = list(X.iloc[[i]]['label'])[0]
            fx = W@xi

            print('xi,yi')
            print(xi,yi)
            sign_fxi = np.sign(fx)

            if yi*sign_fxi <= 0:
                W = W + (alpha*yi*xi)

        file1.write("{},{},{}\n".format(W[0], W[1], W[2]))
        W = list(W)
        
        if W == W_old:
            print('converge')
            convergence = True
            visualize_scatter(X,feat1='f1', feat2='f2', labels='label' , weights=[W[0],W[1],W[2]] , title='')
            break
        else:
            W_old = W

def sign(sum_fxi):
    if sum_fxi <= 0:
        return -1
    else:
        return 1

'''Credit to Kelsey D'Souza'''
def visualize_scatter(df, feat1=0, feat2=1, labels=2, weights=[-1, -1, 1],
                      title=''):
    """
        Scatter plot feat1 vs feat2.
        Assumes +/- binary labels.
        Plots first and second columns by default.
        Args:
          - df: dataframe with feat1, feat2, and labels
          - feat1: column name of first feature
          - feat2: column name of second feature
          - labels: column name of labels
          - weights: [w1, w2, b]
    """

    # Draw color-coded scatter plot
    colors = pd.Series(['r' if label > 0 else 'b' for label in df[labels]])
    ax = df.plot(x=feat1, y=feat2, kind='scatter', c=colors)

    # Get scatter plot boundaries to define line boundaries
    xmin, xmax = ax.get_xlim()

    # Compute and draw line. ax + by + c = 0  =>  y = -a/b*x - c/b
    a = weights[0]
    b = weights[1]
    c = weights[2]

    def y(x):
        return (-a/b)*x - c/b

    line_start = (xmin, xmax)
    line_end = (y(xmin), y(xmax))
    line = mlines.Line2D(line_start, line_end, color='red')
    ax.add_line(line)


    if title == '':
        title = 'Scatter of feature %s vs %s' %(str(feat1), str(feat2))
    ax.set_title(title)

    plt.show()

if __name__ == "__main__":
    main()