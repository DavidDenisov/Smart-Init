import matplotlib.pyplot as plt
import numpy as np

# The code that generates the plots (besides Figure 5) and the values in the tables in the paper.

# Helper function to print_results.
# Given a label (s) and two arrays prints the mean of the arrays (separately) with the label beforehand.
def print_res(s, arr1, arr2,arr3):
    print(s, np.round(np.median(arr1), 3), np.round(np.median(arr2), 3), np.round(np.median(arr3), 3))


# Given array compute the maximum deviation of its values from the mean.
def max_div(arr):
    m = np.mean(arr)
    div = np.abs(arr - m).max()
    return np.round(div, 3)


# Our plot function, recall updating the label between the tests, with how many initial values to skip.
def plot_res(skip=0):
    plt.rcParams['figure.figsize'] = (20, 10)
    plt.rcParams.update({'font.size': 50})
    plt.rcParams.update({'lines.linewidth': 25})

    # Load the results of a test.
    dict = np.load('../results/cifar10/cifar_10.npz')
    values = [dict[l] for l in dict.files]

    n = values[0].shape[2]
    epochs = np.arange(n) + 1

    names = ['Training loss', 'Training accuracy', 'Validation loss', 'Validation accuracy', 'Test loss',
             'Test accuracy']

    # For each value, print the results of the two options compared, with dotted line for the median across the tests
    # and error bars that cover the range of the results.
    for i, v in enumerate(values):
        comp, smart, warm = v[:, 0], v[:, 1], v[:, 2]

        # Plot the results for original.
        s1, s2 = np.percentile(comp,25,axis=0),np.percentile(comp,75,axis=0)
        m_comp = np.median(comp, 0)

        if len(s1)>1:
            plt.plot(epochs[skip:], m_comp[skip:], c='b', label='Original', ls='--')
            plt.fill_between(epochs[skip:], s1[skip:], s2[skip:], alpha=0.25,color='b',edgecolor=None)
        else:
            plt.scatter([1], m_comp, c='b', label='Original',s=5000)
            plt.plot([1, 1], [s1, s2], c='b', label='Original', ls='-', alpha=0.25)

        # Plot the results for Smart-Init.
        s1, s2 = np.percentile(smart,25,axis=0),np.percentile(smart,75,axis=0)
        m_smart = np.median(smart, 0)

        if len(s1)>1:
            plt.plot(epochs[skip:], m_smart[skip:], c='r', label='Smart-init', ls='--')
            plt.fill_between(epochs[skip:], s1[skip:], s2[skip:], alpha=0.25,color='r',edgecolor=None)
        else:
            plt.scatter([1], m_smart, c='r', label='Smart-init',s=5000)
            plt.plot([1, 1], [s1, s2], c='r', label='Smart-init', ls='-', alpha=0.25)

        # Plot the results for warm start.
        s1, s2 = np.percentile(warm,25,axis=0),np.percentile(warm,75,axis=0)
        m_warm = np.median(warm, 0)

        if len(s1)>1:
            plt.plot(epochs[skip:], m_warm[skip:], c='g', label='Warm start', ls='--')
            plt.fill_between(epochs[skip:], s1[skip:], s2[skip:], alpha=0.25,color='g',edgecolor=None)
        else:
            plt.scatter([1], m_warm, c='g', label='Warm start',s=5000)
            plt.plot([1, 1], [s1, s2], c='g', label='Warm start', ls='-', alpha=0.25)

        plt.xlabel('Epoch')
        plt.ylabel(names[i])

        plt.legend()
        # plt.show()

        plt.savefig('../Figs/' + names[i] + '.png', dpi=250, bbox_inches='tight')
        plt.show()


# Print a summary of the results, used to generate the tables.
def print_results():
    # Load the results of a test.
    dict =  np.load('../results/cifar10/cifar_10.npz')
    ratios = [dict[l] for l in dict.files]

    names = ['Training loss', 'Training accuracy', 'Validation loss', 'Validation accuracy', 'Test loss',
             'Test accuracy']
    print('Results order: Original, Smart-Init, warm start.')
    for i, r in enumerate(ratios):
        print(names[i])
        comp, smart, warm = r[:, 0], r[:, 1], r[:, 2]
        if i % 2 == 0:  # If the value is loss, use the lowest value as the best one.
            m1, m2, m3 = np.argmin(comp, 1), np.argmin(smart, 1), np.argmin(warm, 1)
            print_res('Best Epochs:', m1 + 1, m2 + 1, m3 +1)
            a, b, c = np.diag(comp[:, m1]), np.diag(smart[:, m2]), np.diag(warm[:, m3])
            print_res('Best Values:', a, b,c)
            print('s.t.d.', np.round(np.std(a),3), np.round(np.std(b),3), np.round(np.std(c),3))
        else: # If the value is accuracy, use the largest value as the best one.
            m1, m2, m3 = np.argmax(comp, 1), np.argmax(smart, 1), np.argmax(warm, 1)
            print_res('Best Epochs:', m1 + 1, m2 + 1, m3 +1)
            a, b, c = np.diag(comp[:, m1]), np.diag(smart[:, m2]), np.diag(warm[:, m3])
            print_res('Best Values:', a, b, c)
            print('s.t.d.', np.round(np.std(a),3), np.round(np.std(b),3), np.round(np.std(c),3))

if __name__ == '__main__':
    print_results()
    plot_res(25)