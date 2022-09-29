import math
import os
import random
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import ConvexHull
from kneed import KneeLocator
import matplotlib.pyplot as plt
import numpy as np


def generate_point(mean_x, mean_y, deviation_x, deviation_y):
    return random.gauss(mean_x, deviation_x), random.gauss(mean_y, deviation_y)


def generate_data(number_of_clusters=5, points_per_cluster=50, cluster_mean_x=100, cluster_mean_y=100,
                  cluster_deviation_x=50, cluster_deviation_y=50, point_deviation_x=5, point_deviation_y=5):

    cluster_centers = [generate_point(cluster_mean_x,
                                      cluster_mean_y,
                                      cluster_deviation_x,
                                      cluster_deviation_y)
                       for i in range(number_of_clusters)]

    data = [generate_point(center_x,
                           center_y,
                           point_deviation_x,
                           point_deviation_y)
            for center_x, center_y in cluster_centers
            for i in range(points_per_cluster)]

    plt.clf()
    plt.scatter(*zip(*data))
    plt.savefig('Unclustered Data')

    return data


def cluster_data(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(data)
    plt.clf()
    plt.scatter(*zip(*data), c=kmeans.labels_)
    centroids = kmeans.cluster_centers_
    for cent in centroids:
        plt.scatter(cent[0], cent[1], c='black', marker='x')
    plt.savefig('Clustered data for ' + str(num_clusters) + ' clusters')
    return kmeans, plt


def get_variance(km):
    return km.inertia_


def plot_knee_graph(var_lst, k_range):
    kn = KneeLocator(k_range, var_lst, curve='convex', direction='decreasing')
    plt.clf()
    plt.xlabel('Number of clusters k')
    plt.ylabel('Sum of squared distances')
    plt.plot(k_range, var_lst, 'bx-')
    plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.savefig('Knee graph')


def get_silhouette_score(km, data):
    km.predict(data)
    return silhouette_score(data, km.labels_, metric='euclidean')


def plot_silhouette_graph(sil_lst, k_range):
    plt.clf()
    plt.xlabel('Number of clusters k')
    plt.ylabel('Average silhouette score')
    plt.plot(k_range, sil_lst, 'bx-')
    max_silhouette_score = k_range[sil_lst.index(max(sil_lst))]
    plt.vlines(max_silhouette_score, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.savefig('Silhouette graph')

def plot_chtc_graph(chtc_lst, k_range):
    plt.clf()
    plt.xlabel('Number of clusters k')
    plt.ylabel('Average CHTC score')
    plt.plot(k_range, chtc_lst, 'bx-')
    max_chtc_score = k_range[chtc_lst.index(max(chtc_lst))]
    plt.vlines(max_chtc_score, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.savefig('CHTC graph')


def get_clusters(km, data):
    clstrs = []
    cntrs = []
    for i in range(km.n_clusters):
        clstrs.append([data[j] for j in range(len(km.labels_)) if km.labels_[j] == i])
        cntrs.append(km.cluster_centers_[i])
    return clstrs, cntrs


def get_convex_hull_points(cluster_points):
    points = np.array(cluster_points)
    hull = ConvexHull(points)
    return [tuple(t) for t in points[hull.vertices]]


def cluster_indices(cluster_num, labels_array):
    return np.where(labels_array == cluster_num)[0]


def get_chtc_score_for_hull_point(hull_point, cents):
    dists = []
    for c in cents:
        dists.append(math.dist(hull_point, c))
    dist_to_cent = min(dists)
    dists.remove(dist_to_cent)
    dist_to_closest_cent = min(dists)
    return (dist_to_closest_cent - dist_to_cent) / max(dist_to_cent, dist_to_closest_cent)


def get_chtc_score(km, data, plot):
    clusters, centers = get_clusters(km, data)
    clusters_chtc = []
    for c in clusters:
        plot_cluster_hull(c, plot)
        hull_chtc = []
        hull_points = get_convex_hull_points(c)
        for p in hull_points:
            hull_chtc.append(get_chtc_score_for_hull_point(p, centers))
        clusters_chtc.append(sum(hull_chtc)/len(hull_points))
    plt.savefig('Convex hulls graph for ' + str(len(clusters)) + ' clusters')
    plt.clf()
    return sum(clusters_chtc)/len(clusters)


def plot_cluster_hull(cluster, plt):
    points = np.array(cluster)
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'c')
    plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'o', mec='r', color='none', lw=1, markersize=10)
    return plt


def evaluate_k(data, k):
    km, plot = cluster_data(data, k)
    variance = get_variance(km)
    silhouette = get_silhouette_score(km, data)
    chtc = get_chtc_score(km, data, plot)
    return variance, silhouette, chtc


def experiment():
    date_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    os.mkdir(date_time)
    os.chdir(date_time)

    data = generate_data()
    variance_lst = []
    silhouette_lst = []
    chtc_lst = []
    for k in range(2, 10):
        var_score, sil_score, chtc_score = (evaluate_k(data, k))
        variance_lst.append(var_score)
        silhouette_lst.append(sil_score)
        chtc_lst.append(chtc_score)
    plot_knee_graph(variance_lst, range(2, 10))
    plot_silhouette_graph(silhouette_lst, range(2, 10))
    plot_chtc_graph(chtc_lst, range(2, 10))
    os.chdir("..")
    return variance_lst, silhouette_lst, chtc_lst


def get_indices_by_largest_element(scores_lst):
    return np.array(scores_lst).argsort().tolist()[::-1]


def k_distance(sil_lst, chtc_lst):
    distance = 0
    for i in range(len(chtc_lst)):
        for j in range(len(chtc_lst)):
            if sil_lst[j] == chtc_lst[i]:
                distance += abs(i-j)
                break
    return distance


def k_distance_accuracy(sil_lst, chtc_lst):
    max_dist = k_distance(chtc_lst, chtc_lst[::-1])
    dist = k_distance(sil_lst, chtc_lst)
    return (max_dist - dist) / max_dist


def plot_generic_graph(graph_name, x_label, x_values_lst, y_label, y_values_lst):
    plt.clf()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x_values_lst, y_values_lst, 'bx-')
    plt.savefig(graph_name)


def experiments(num_experiments):
    chtc_sil_best_k_correct = 0
    chtc_sil_k_dist_acc = 0
    chtc_var_best_k_correct = 0
    for i in range(num_experiments):
        var, sil, chtc = experiment()
        kn = KneeLocator(range(2, 10), var, curve='convex', direction='decreasing')
        var_ind = kn.knee - 2
        sil_ind = get_indices_by_largest_element(sil)
        chtc_ind = get_indices_by_largest_element(chtc)
        if chtc_ind[0] == sil_ind[0]:
            chtc_sil_best_k_correct += 1
        if chtc_ind[0] == var_ind:
            chtc_var_best_k_correct += 1
        chtc_sil_k_dist_acc += k_distance_accuracy(sil_ind, chtc_ind)
    print("CHTC to silhouette best K accuracy: " + str(chtc_sil_best_k_correct/num_experiments))
    print("CHTC to knee best K accuracy: " + str(chtc_var_best_k_correct/num_experiments))
    print("CHTC to silhouette every K accuracy: " + str(chtc_sil_k_dist_acc/num_experiments))


if __name__ == "__main__":
    experiments(100)








