#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
    // параметры для приближенного поиска
    index_params = flannKdTreeIndexParams(4);
    search_params = flannKsTreeSearchParams(32);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    cv::Mat ids, dists;
    flann_index->knnSearch(query_desc, ids, dists, k, *search_params);
    for (int i = 0; i < query_desc.rows; i++) {
        std::vector<cv::DMatch> matches_i(k);
        for (int j = 0; j < k; j++)
            matches_i[j] = cv::DMatch(i, ids.at<int>(i, j), dists.at<float>(i, j));
        matches.push_back(matches_i);
    }
}
