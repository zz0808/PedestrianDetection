#include <iostream>
#include <fstream>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <io.h>
#include <direct.h>
#include "detection.h"
#include "pretreated.h"

int main() {

	if (DEALIMAGE) 
		int flag = deal_image();

	if (CROP)
		int flag = crop_image();

	//检测窗口(64,128),块尺寸(16,16),块步长(8,8),cell 尺寸(8,8),直方图bin个数9
	cv::HOGDescriptor hog(cv::Size(64, 128), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);
	int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();// 创建分类器


	if (TRAIN) { // 是否重新训练
		svm_train(hog, DescriptorDim, svm);

	} else {
		svm = cv::ml::SVM::load(SvmListFile);
	}

	std::vector<float> vec;

	std::cout << "loaded SVM_HOG.xml file" << std::endl;

	int svdim = svm->getVarCount(); // 特征向量的维数,即hog的描述子维数
	cv::Mat svecsmat = svm->getSupportVectors();
	int numofsv = svecsmat.rows; // 支持向量的个数

	std::cout << "the number of support vectors: " << numofsv << std::endl;

	cv::Mat alphaMat = cv::Mat::zeros(1, numofsv, CV_32FC1); // alpha 向量
	cv::Mat supportVectorMat = cv::Mat::zeros(numofsv, svdim, CV_32FC1); // 支持向量矩阵
	cv::Mat resultMat = cv::Mat::zeros(1, svdim, CV_32FC1); // alpha 向量乘以支持向量矩阵的结果

	std::vector<float> alpha_vec, sv_vec;
	supportVectorMat = svecsmat;

	//将alpha向量的数据复制到alphaMat中
	double rho = svm->getDecisionFunction(0, alpha_vec, sv_vec);

	for (size_t i = 0; i < numofsv; i++) {
		alphaMat.at<float>(0, i) = alpha_vec[i];
	}

	std::cout << "alphaMat size: " << alphaMat.size() << std::endl;
	std::cout << "supportVectorMat size: " << supportVectorMat.size() << std::endl;
	resultMat = -1 * alphaMat * supportVectorMat;

	// 将resultMat矩阵的结果复制到向量vec中
	for (int i = 0; i < svdim; i++) {
		vec.push_back(resultMat.at<float>(0, i));
	}

	vec.push_back(static_cast<float>(rho));

	std::cout << "going to write the HOGDetectorForOpenCV.txt file" << std::endl;

	std::ofstream fout(HogDetectorListFile);
	for (int i = 0; i < vec.size(); i++) {
		fout << vec[i] << std::endl;
	}
	fout.close();



	// --------------------------------------测试-----------------------------------------//
	cv::HOGDescriptor hog_test;
	hog_test.setSVMDetector(vec);

	cv::Mat frame = cv::imread("Test4.jpg");

	std::vector<cv::Rect> found, found_filtered;
	hog_test.detectMultiScale(frame, found, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 2);

	//找出所有没有嵌套的矩形框r,并放入found_filtered中,如果有嵌套的话,则取外面最大的那个矩形框放入found_filtered中
	for (int i = 0; i < found.size(); i++)
	{
		cv::Rect r = found[i];
		int j = 0;
		for (; j < found.size(); j++)
			if (j != i && (r & found[j]) == r)
				break;
		if (j == found.size())
			found_filtered.push_back(r);
	}

	//画矩形框，因为hog检测出的矩形框比实际人体框要稍微大些,所以这里需要做一些调整
	for (int i = 0; i < found_filtered.size(); i++)
	{
		cv::Rect r = found_filtered[i];
		r.x += cvRound(r.width * 0.1);
		r.width = cvRound(r.width * 0.8);
		r.y += cvRound(r.height * 0.07);
		r.height = cvRound(r.height * 0.8);
		rectangle(frame, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);
	}

	//cv::imwrite("ImgProcessed.jpg", src);
	cv::namedWindow("Test");
	cv::imshow("Test", frame);


	cv::waitKey(0);

	return 0;
}