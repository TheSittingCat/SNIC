from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
        
    # Description: The raw results from the statistical analysis of the zero-shot learning models.
    norm1 = [73.9, 71.1, 84.5, 93.1, 7.1, 66.7, 73.1, 83.3, 83.5, 87.9, 91.9, 73]
    norm2 = [76.1, 69.5, 61.6, 47.2, 0.7, 62.9, 49.9, 68.3, 49.3, 56, 64.5, 72.1]
    norm3 = [21.22, 27.02, 56.05, 96.99, 3.7, 21.82, 35.53, 59.45, 92.19, 73.47, 96.49, 34.13]
    norm4 = [24.8, 28.5, 28.2, 27.2, 9.6, 23.8, 33.0, 24.4, 54.8, 32.2, 45.8, 30.8]
    norm5 = [16.5, 31.2, 15.9, 81.2, 4.1, 37.3, 31.6, 16.7, 17.6, 42.4, 68.2, 25.6]
    norm6 = [69.1, 54.9, 25.9, 33.5, 16.1, 69.2, 55.9, 56.2, 30.4, 53.2, 60.6, 28.2]
    norm7 = [50.8, 55.8, 19.8, 30.1, 16.6, 51.4, 56.0, 41.6, 28.0, 41.4, 36.6, 25.1]
    #c1 = [24.9, 16.9, 22.9, 6.0, 26.2, 17.9, 22.7, 23.6, 17.1, 16.6]
    c1 = [15.6, 11.7, 16.9, 13.0, 16.6, 24.9, 17.2, 9.3, 21.3, 16.9, 18.2, 16.2]
    c2 = [71.9, 64.9, 29.3, 41.7, 33.6, 61.1, 54.4, 50.07, 28.5, 59.6, 54, 43.5]

    norm1_2 = [68.4, 72.9, 93.2, 92.1, 77.6, 74.0, 80.1, 74.1, 67.3]
    norm2_2 = [37.3, 87.8, 51.2, 42.3, 61.4, 55.4, 66.1, 73.4, 71.7]
    norm3_2 = [19.2, 21.62, 72.87, 98.19, 64.46, 50.15, 49.14, 34.37, 58.75]
    norm4_2 = [29.8, 58.9, 20.06, 40.8, 36.1, 34.5, 33.3, 22.8, 73.5]
    norm5_2 = [15.8, 21.9, 13.9, 57.5, 5.1, 38.7, 57.8, 24.3, 7.3]
    norm6_2 = [26.8, 53.9, 39.7, 48.6, 20.3, 31.9, 32.1, 85.6, 40.4]
    norm7_2 = [16.6, 48.9, 31.8, 38.6, 18.6, 26.7, 27.6, 80.3, 32.7]
    c1_2 = [11.7, 22.9, 12.9, 28.4, 11.2, 19.3, 8.3, 28.9, 21.4]
    c2_2 = [22.1, 41.1, 29.3, 36.0, 42.3, 43.2, 45.0, 62.5, 30.1]

    results_combined = [norm1, norm2, norm3, norm4, norm5, norm6, norm7, c1, c2]
    results_combined_2 = [norm1_2, norm2_2, norm3_2, norm4_2, norm5_2, norm6_2, norm7_2, c1_2, c2_2]

    data = np.empty((len(results_combined), len(results_combined))) # Create a matrix to store the spearman correlation values
    for i in range(len(results_combined)):
        for j in range(len(results_combined)):
            data[i,j] = (spearmanr(results_combined[i], results_combined[j]).correlation)
    zero_buttom = np.tri(data.shape[0], data.shape[1], k = -1)
    data = np.ma.array(data, mask = zero_buttom)


    fig, ax = plt.subplots(figsize=(10, 10))
    #plt.figure(figsize=(10, 10))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if isinstance(data[i, j], np.ma.core.MaskedConstant):
                continue
            ax.text(j, i, round(float(data[i, j]), 2), ha='center', va='center', color='black')
    plt.title("Spearman Correlation Matrix of Differnt Norms Across Models")
    plt.imshow(data, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(results_combined)), ["norm1", "norm2", "norm3", "norm4", "norm5", "norm6", "norm7", "c1", "c2"])
    plt.yticks(range(len(results_combined)), ["norm1", "norm2", "norm3", "norm4", "norm5", "norm6", "norm7", "c1", "c2"])
    plt.savefig("spearman_correlation_matrix.png")
    print("Done")

    for performance in results_combined:
        print("Mean:" + str(results_combined.index(performance)+ 1) + " " + str(np.mean(performance)))
    
    print("Moving on to Prompt Only \n")
    
    for performance in results_combined_2:
        print("Mean:" + str(results_combined_2.index(performance)+ 1) + " " + str(np.mean(performance)))