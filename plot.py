import matplotlib.pyplot as plt
import numpy as np

# The code that generates the plots (besides Figure 5) and the values in the tables in the paper.

# Helper function to print_results.
# Given a label (s) and two arrays prints the mean of the arrays (separately) with the label beforehand.
def print_res(s, arr1, arr2):
    print(s, np.round(np.mean(arr1), 3), np.round(np.mean(arr2), 3))


# Given array compute the maximum deviation of its values from the mean.
def max_div(arr):
    m = np.mean(arr)
    div = np.abs(arr - m).max()
    return np.round(div, 3)


# Our plot function, recall updating the label between the tests, with how many initial values to skip.
def plot_res(skip=0):
    plt.rcParams['figure.figsize'] = (16, 9)
    plt.rcParams.update({'font.size': 50})
    plt.rcParams.update({'lines.linewidth': 25})

    # Load the results of a test.
    dict = np.load('image_64.npz')
    ratios = [dict[l] for l in dict.files]

    n = ratios[0].shape[2]
    epochs = np.arange(n) + 1

    names = ['Training loss', 'Training accuracy', 'Validation loss', 'Validation accuracy', 'Test loss',
             'Test accuracy']

    # For each value, print the results of the two options compared, with dotted line for the median across the tests
    # and error bars that cover the range of the results.
    for i, r in enumerate(ratios):
        comp, our = r[:, 0], r[:, 1]
        # Plot the results for option 1, recall updating the label between the tests.
        s1, s2 = np.min(comp, 0), np.max(comp, 0)
        m_comp = np.median(comp, 0)

        if len(s1)>1:
            plt.plot(epochs[skip:], m_comp[skip:], c='b', label='Original', ls='--')
            plt.fill_between(epochs[skip:], s1[skip:], s2[skip:], alpha=0.25,color='b',edgecolor=None)
        else:
            plt.scatter([1], m_comp, c='b', label='Original',s=5000)
            plt.plot([1, 1], [s1, s2], c='b', label='Original', ls='-', alpha=0.25)

        # Plot the results for option 2, recall updating the label between the tests.
        s1, s2 = np.min(our, 0), np.max(our, 0)
        m_our = np.median(our, 0)

        if len(s1) > 1:
            plt.plot(epochs[skip:], m_our[skip:], c='r', label='Smart-init', ls='--')
            plt.fill_between(epochs[skip:], s1[skip:], s2[skip:], alpha=0.25, color='r', edgecolor=None)
        else:
            plt.scatter([1], m_our, c='r', label='Smart-init',s=5000)
            plt.plot([1,1], [s1, s2], c='r', label='Smart-init', ls='-', alpha=0.25)
        # plt.errorbar(epochs[skip:], m_our[skip:], s_our[:, skip:], c='r', label='Smart Init', ls='--', marker='s',
        #              mfc='r')
        # plt.errorbar(epochs[skip:], m_comp[skip:], s_comp[:, skip:], c='b', label='Original', ls='--', marker='s',
        #              mfc='b')

        plt.xlabel('Epoch')
        plt.ylabel(names[i])

        plt.legend()
        # plt.show()

        plt.savefig('./Figs/' + names[i] + '.png', dpi=250, bbox_inches='tight')
        plt.show()


# Print a summary of the results, used to generate the tables.
def print_results():
    # Load the results of a test.
    dict = np.load('image_16.npz')
    ratios = [dict[l] for l in dict.files]

    names = ['Training loss', 'Training accuracy', 'Validation loss', 'Validation accuracy', 'Test loss',
             'Test accuracy']
    for i, r in enumerate(ratios):
        # print(len(r[:, 0][0]))
        print(names[i])
        comp, our = r[:, 0], r[:, 1]
        if i % 2 == 0:  # If the value is loss, use the lowest value as the best one.
            m1, m2 = np.argmin(comp, 1), np.argmin(our, 1)
            print_res('Best Epochs:', m1 + 1, m2 + 1)
            a, b = np.diag(comp[:, m1]), np.diag(our[:, m2])
            print_res('Best Values:', a, b)
            print('s.t.d.', np.round(np.std(a),3), np.round(np.std(b),3))
        else: # If the value is accuracy, use the largest value as the best one.
            m1, m2 = np.argmax(comp, 1), np.argmax(our, 1)
            print_res('Best Epochs:', m1 + 1, m2 + 1)
            a, b = np.diag(comp[:, m1]), np.diag(our[:, m2])
            print_res('Best Values:', a, b)
            print('s.t.d.', np.round(np.std(a),3), np.round(np.std(b),3))


if __name__ == '__main__':
    print_results()
    # plot_res(1)