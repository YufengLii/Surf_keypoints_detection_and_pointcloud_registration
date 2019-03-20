#include <iostream>  
#include <stdio.h>  
#include <vector>

#include "opencv2/core.hpp"  
#include "opencv2/core/utility.hpp"  
#include "opencv2/core/ocl.hpp"  
#include "opencv2/imgcodecs.hpp"  
#include "opencv2/highgui.hpp"  
#include "opencv2/features2d.hpp"  
#include "opencv2/calib3d.hpp"  
#include "opencv2/imgproc.hpp"  
#include "opencv2/flann.hpp"  
#include "opencv2/xfeatures2d.hpp"  
#include "opencv2/ml.hpp"  

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;
using namespace cv::ml;

vector<KeyPoint> Ransac_keypoint1, Ransac_keypoint2;
vector<DMatch> Ransac_matches;

Mat FundamentalRansac(vector<KeyPoint> &current_keypoint1, vector<KeyPoint> &current_keypoint2, vector<DMatch> &current_matches) {

	vector<Point2f>p1, p2;
	for (size_t i = 0; i < current_matches.size(); i++)
	{
		p1.push_back(current_keypoint1[i].pt);
		p2.push_back(current_keypoint2[i].pt);
	}

	vector<uchar> RansacStatus;
	Mat Fundamental = findFundamentalMat(p1, p2, RansacStatus, FM_RANSAC);
	int index = 0;
	for (size_t i = 0; i < current_matches.size(); i++)
	{
		if (RansacStatus[i] != 0)
		{
			Ransac_keypoint1.push_back(current_keypoint1[i]);
			Ransac_keypoint2.push_back(current_keypoint2[i]);
			current_matches[i].queryIdx = index;
			current_matches[i].trainIdx = index;
			Ransac_matches.push_back(current_matches[i]);
			index++;
		}
	}
	return Fundamental;
}





int main(int argc, char** argv)
{
	Mat a = imread(argv[1], IMREAD_GRAYSCALE);
	Mat b = imread(argv[2], IMREAD_GRAYSCALE);

	Ptr<SURF> surf;
	surf = SURF::create(800);
	BFMatcher matcher;
	Mat c, d;
	vector<KeyPoint>key1, key2;
	vector<DMatch> matches;

	surf->detectAndCompute(a, Mat(), key1, c);
	surf->detectAndCompute(b, Mat(), key2, d);



	matcher.match(c, d, matches);
	sort(matches.begin(), matches.end());

	cout << "总匹配点数为: " << matches.size() << endl << endl;
	cout << "初始选择排序靠前的  " << (int)(matches.size() / atoi(argv[3])) << "个，进行Ransac." << endl << endl;
	vector< DMatch > good_matches;

	int ptsPairs = std::min((int)(matches.size() / atoi(argv[3])), (int)matches.size());
	for (int i = 0; i < ptsPairs; i++)
	{
		good_matches.push_back(matches[i]);
	}

	vector<KeyPoint> keypoint1, keypoint2;
	for (size_t i = 0; i < good_matches.size(); i++)
	{
		keypoint1.push_back(key1[good_matches[i].queryIdx]);
		keypoint2.push_back(key2[good_matches[i].trainIdx]);
	}



	Mat img_matches;
	drawMatches(a, key1, b, key2, good_matches, img_matches);
	namedWindow("误匹配消除前", CV_WINDOW_NORMAL);
	imshow("误匹配消除前", img_matches);
	cvWaitKey(1);


	int times = 0, current_num = 1, per_num = 0;;
	Mat img_Ransac_matches;
	char window_name[] = "0次Ransac之后匹配结果";
	Mat Fundamental;
	while (1) {
		if (per_num != current_num) {
			Ransac_keypoint1.clear();
			Ransac_keypoint2.clear();
			Ransac_matches.clear();
			per_num = good_matches.size();
			Fundamental = FundamentalRansac(keypoint1, keypoint2, good_matches);
			cout << endl << "Ransac" << ++times << "次之后的匹配点数为：" << Ransac_matches.size() << endl;
			cout << "基础矩阵:" << endl;
			cout << Fundamental << endl << endl << endl;
			window_name[0] = times + '0';
			drawMatches(a, Ransac_keypoint1, b, Ransac_keypoint2, Ransac_matches, img_Ransac_matches);
			namedWindow(window_name, CV_WINDOW_NORMAL);
			imshow(window_name, img_Ransac_matches);
			cvWaitKey(1);
			keypoint1.clear();
			keypoint2.clear();
			good_matches.clear();
			keypoint1 = Ransac_keypoint1;
			keypoint2 = Ransac_keypoint2;
			good_matches = Ransac_matches;
			current_num = good_matches.size();
		}
		else
			break;

	}
	cvWaitKey(0);
	return 0;
}