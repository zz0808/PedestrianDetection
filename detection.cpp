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

	//��ⴰ��(64,128),��ߴ�(16,16),�鲽��(8,8),cell �ߴ�(8,8),ֱ��ͼbin����9
	cv::HOGDescriptor hog(cv::Size(64, 128), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);
	int DescriptorDim;//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();// ����������


	if (TRAIN) { // �Ƿ�����ѵ��
		svm_train(hog, DescriptorDim, svm);

	} else {
		svm = cv::ml::SVM::load(SvmListFile);
	}

	std::vector<float> vec;

	std::cout << "loaded SVM_HOG.xml file" << std::endl;

	int svdim = svm->getVarCount(); // ����������ά��,��hog��������ά��
	cv::Mat svecsmat = svm->getSupportVectors();
	int numofsv = svecsmat.rows; // ֧�������ĸ���

	std::cout << "the number of support vectors: " << numofsv << std::endl;

	cv::Mat alphaMat = cv::Mat::zeros(1, numofsv, CV_32FC1); // alpha ����
	cv::Mat supportVectorMat = cv::Mat::zeros(numofsv, svdim, CV_32FC1); // ֧����������
	cv::Mat resultMat = cv::Mat::zeros(1, svdim, CV_32FC1); // alpha ��������֧����������Ľ��

	std::vector<float> alpha_vec, sv_vec;
	supportVectorMat = svecsmat;

	//��alpha���������ݸ��Ƶ�alphaMat��
	double rho = svm->getDecisionFunction(0, alpha_vec, sv_vec);

	for (size_t i = 0; i < numofsv; i++) {
		alphaMat.at<float>(0, i) = alpha_vec[i];
	}

	std::cout << "alphaMat size: " << alphaMat.size() << std::endl;
	std::cout << "supportVectorMat size: " << supportVectorMat.size() << std::endl;
	resultMat = -1 * alphaMat * supportVectorMat;

	// ��resultMat����Ľ�����Ƶ�����vec��
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



	// --------------------------------------����-----------------------------------------//
	cv::HOGDescriptor hog_test;
	hog_test.setSVMDetector(vec);

	cv::Mat frame = cv::imread("Test4.jpg");

	std::vector<cv::Rect> found, found_filtered;
	hog_test.detectMultiScale(frame, found, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 2);

	//�ҳ�����û��Ƕ�׵ľ��ο�r,������found_filtered��,�����Ƕ�׵Ļ�,��ȡ���������Ǹ����ο����found_filtered��
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

	//�����ο���Ϊhog�����ľ��ο��ʵ�������Ҫ��΢��Щ,����������Ҫ��һЩ����
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